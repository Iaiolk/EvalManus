import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection

TOOL_CALL_REQUIRED = "需要工具调用但未提供"


class ToolCallAgent(ReActAgent):
    """用于处理工具/函数调用的基础代理类，具有增强的抽象能力"""

    name: str = "toolcall"
    description: str = "一个可以执行工具调用的代理。"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """使用工具处理当前状态并决定下一步行动"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 获取带工具选项的响应
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # 检查这是否是包含TokenLimitExceeded的RetryError
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(f"🚨 令牌限制错误 (来自RetryError): {token_limit_error}")
                self.memory.add_message(
                    Message.assistant_message(
                        f"达到最大令牌限制，无法继续执行: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # 记录响应信息
        logger.info(f"✨ {self.name}的思考: {content}")
        logger.info(
            f"🛠️ {self.name}选择了{len(tool_calls) if tool_calls else 0}个工具来使用"
        )
        if tool_calls:
            logger.info(
                f"🧰 准备使用的工具: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 工具参数: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("未从LLM收到响应")

            # 处理不同的tool_choices模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(f"🤔 嗯，{self.name}尝试使用工具，但工具不可用！")
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 将在act()中处理

            # 对于'auto'模式，如果没有命令但有内容则继续
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 哎呀！{self.name}的思考过程遇到了一个问题: {e}")
            self.memory.add_message(
                Message.assistant_message(f"处理过程中遇到错误: {str(e)}")
            )
            return False

    async def act(self) -> str:
        """执行工具调用并处理其结果"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # 如果没有工具调用则返回最后一条消息内容
            return self.messages[-1].content or "没有内容或命令可执行"

        results = []
        for command in self.tool_calls:
            # 为每个工具调用重置base64_image
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(f"🎯 工具'{command.function.name}'完成了任务！结果: {result}")

            # 将工具响应添加到内存
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用，具有强大的错误处理能力"""
        if not command or not command.function or not command.function.name:
            return "错误: 无效的命令格式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误: 未知工具'{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 激活工具: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 处理特殊工具
            await self._handle_special_tool(name=name, result=result)

            # 检查结果是否是带有base64_image的ToolResult
            if hasattr(result, "base64_image") and result.base64_image:
                # 存储base64_image以供稍后在tool_message中使用
                self._current_base64_image = result.base64_image

            # 格式化结果以供显示（标准情况）
            observation = (
                f"观察到执行的命令`{name}`的输出:\n{str(result)}"
                if result
                else f"命令`{name}`完成，没有输出"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"解析{name}的参数时出错: 无效的JSON格式"
            logger.error(
                f"📝 哎呀！'{name}'的参数不合理 - 无效的JSON, 参数:{command.function.arguments}"
            )
            return f"错误: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ 工具'{name}'遇到了问题: {str(e)}"
            logger.exception(error_msg)
            return f"错误: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行和状态变化"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 设置代理状态为已完成
            logger.info(f"🏁 特殊工具'{name}'已完成任务！")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """确定工具执行是否应该结束代理"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """清理代理工具使用的资源。"""
        logger.info(f"🧹 清理代理'{self.name}'的资源...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 清理工具: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(f"🚨 清理工具'{tool_name}'时出错: {e}", exc_info=True)
        logger.info(f"✨ 代理'{self.name}'的清理完成。")

    async def run(self, request: Optional[str] = None) -> str:
        """运行代理并在完成后进行清理。"""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
