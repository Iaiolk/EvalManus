import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """定义计划步骤可能状态的枚举类"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """返回所有可能的步骤状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """返回表示活动状态（未开始或进行中）的值列表"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """返回状态到其标记符号的映射"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """使用代理管理任务规划和执行的流程类"""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 在调用super().__init__之前设置执行器键
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 如果提供了计划ID，则设置
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 如果未提供planning_tool，则初始化一个
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 使用处理后的数据调用父类的初始化方法
        super().__init__(agents, **data)

        # 如果未指定executor_keys，则设置为所有代理键
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        获取当前步骤的适当执行器代理。
        可以扩展以根据步骤类型/要求选择代理。
        """
        # 如果提供了步骤类型并且与代理键匹配，则使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则使用第一个可用的执行器或回退到主要代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 回退到主要代理
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """使用代理执行计划流程。"""
        try:
            if not self.primary_agent:
                raise ValueError("没有可用的主代理")

            # 如果提供了输入，则创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否创建成功
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"计划创建失败。计划ID {self.active_plan_id} 在规划工具中未找到。"
                    )
                    return f"为以下内容创建计划失败: {input_text}"

            result = ""
            while True:
                # 获取要执行的当前步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划已完成，则退出
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用适当的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                logger.info(f"🤖 excutor: {executor.name}, state: {executor.state}")
                # 检查代理是否想要终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"PlanningFlow中出错: {str(e)}")
            return f"执行失败: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """使用流程的LLM和PlanningTool基于请求创建初始计划。"""
        logger.info(f"正在创建ID为{self.active_plan_id}的初始计划")

        # 为计划创建创建系统消息
        system_message = Message.system_message(
            "你是一个规划助手。创建一个简洁、可操作的计划，包含清晰的步骤。"
            "专注于关键里程碑而不是详细的子步骤。"
            "优化清晰度和效率。"
        )

        # 创建包含请求的用户消息
        user_message = Message.user_message(
            f"创建一个合理的计划，包含清晰的步骤来完成任务: {request}"
        )

        # 使用PlanningTool调用LLM
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 如果存在工具调用，则处理
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"解析工具参数失败: {args}")
                            continue

                    # 确保plan_id设置正确并执行工具
                    args["plan_id"] = self.active_plan_id

                    # 通过ToolCollection执行工具而不是直接执行
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"计划创建结果: {str(result)}")
                    return

        # 如果执行到这里，创建默认计划
        logger.warning("正在创建默认计划")

        # 使用ToolCollection创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"计划: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["分析请求", "执行任务", "验证结果"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        解析当前计划以识别第一个未完成步骤的索引和信息。
        如果没有找到活动步骤，则返回(None, None)。
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"未找到ID为{self.active_plan_id}的计划")
            return None, None

        try:
            # 从规划工具存储中直接访问计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤类型/类别（如果可用）
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型（例如，[SEARCH]或[CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"将步骤标记为进行中时出错: {e}")
                        # 如果需要，直接更新步骤状态
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 未找到活动步骤

        except Exception as e:
            logger.warning(f"查找当前步骤索引时出错: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """使用agent.run()通过指定的代理执行当前步骤。"""
        # 为代理准备当前计划状态的上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 创建一个提示，让代理执行当前步骤
        step_prompt = f"""
        当前计划状态:
        {plan_status}

        您当前的任务:
        您现在正在处理步骤 {self.current_step_index}: "{step_text}"

        请使用适当的工具执行此步骤。完成后，提供一个关于您完成了什么的摘要。
        """

        # 使用agent.run()执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 成功执行后将步骤标记为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index} 时出错: {e}")
            return f"执行步骤 {self.current_step_index} 时出错: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """将当前步骤标记为已完成。"""
        if self.current_step_index is None:
            return

        try:
            # 将步骤标记为已完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"已将计划 {self.active_plan_id} 中的步骤 {self.current_step_index} 标记为已完成"
            )
        except Exception as e:
            logger.warning(f"更新计划状态失败: {e}")
            # 在规划工具存储中直接更新步骤状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保step_statuses列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本。"""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"获取计划时出错: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """如果规划工具失败，直接从存储中生成计划文本。"""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"错误：未找到ID为{self.active_plan_id}的计划"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "未命名计划")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保step_statuses和step_notes与步骤数量匹配
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 按状态统计步骤数
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"计划: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += f"进度: {completed}/{total} 步骤已完成 ({progress:.1f}%)\n"
            plan_text += f"状态: {status_counts[PlanStepStatus.COMPLETED.value]} 已完成, {status_counts[PlanStepStatus.IN_PROGRESS.value]} 进行中, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} 已阻塞, {status_counts[PlanStepStatus.NOT_STARTED.value]} 未开始\n\n"
            plan_text += "步骤:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态标记来指示步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   备注: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"从存储生成计划文本时出错: {e}")
            return f"错误: 无法检索ID为{self.active_plan_id}的计划"

    async def _finalize_plan(self) -> str:
        """完成计划并使用流程的LLM直接提供摘要。"""
        plan_text = await self._get_plan_text()

        # 使用流程的LLM直接创建摘要
        try:
            system_message = Message.system_message(
                "你是一个规划助手。你的任务是总结已完成的计划。"
            )

            user_message = Message.user_message(
                f"计划已完成。以下是最终的计划状态：\n\n{plan_text}\n\n请提供已完成工作的摘要和任何最终想法。"
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"计划已完成：\n\n{response}"
        except Exception as e:
            logger.error(f"使用LLM完成计划时出错: {e}")

            # 回退到使用代理生成摘要
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                计划已完成。以下是最终的计划状态：

                {plan_text}

                请提供已完成工作的摘要和任何最终想法。
                """
                summary = await agent.run(summary_prompt)
                return f"计划已完成：\n\n{summary}"
            except Exception as e2:
                logger.error(f"使用代理完成计划时出错: {e2}")
                return "计划已完成。生成摘要时出错。"
