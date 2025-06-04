"""
Manus工作流执行器
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from app.agent.base import BaseAgent
from app.logger import logger
from app.workflow.base import (
    BaseWorkflow,
    WorkflowDefinition,
    WorkflowExecutionContext,
    WorkflowNode,
    WorkflowState,
)


class ManusWorkflowExecutor(BaseWorkflow):
    """Manus风格的工作流执行器"""

    # Agent注册表，子类需要重写
    agent_registry: Dict[str, Type[BaseAgent]] = {}

    def __init__(
        self, definition: WorkflowDefinition, execution_id: Optional[str] = None
    ):
        execution_id = execution_id or f"exec_{uuid.uuid4().hex[:8]}"
        context = WorkflowExecutionContext(
            workflow_id=definition.id, execution_id=execution_id
        )
        super().__init__(definition=definition, context=context)

        # 验证工作流定义
        self.definition.validate_dependencies()

    async def execute(self) -> Dict[str, Any]:
        """执行工作流"""
        logger.info(f"开始执行工作流: {self.definition.name}")
        self.context.state = WorkflowState.RUNNING
        self.context.start_time = datetime.now()

        try:
            # 获取执行顺序
            execution_levels = self.definition.get_execution_order()

            for level_nodes in execution_levels:
                # 并行执行同一层级的节点
                await self._execute_level(level_nodes)

                # 检查是否被暂停或取消
                if self.context.state in [
                    WorkflowState.PAUSED,
                    WorkflowState.CANCELLED,
                ]:
                    break

            if self.context.state == WorkflowState.RUNNING:
                self.context.state = WorkflowState.COMPLETED

        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            self.context.state = WorkflowState.FAILED
            self.context.error_info = {
                "error": str(e),
                "node": self.context.current_node,
                "timestamp": datetime.now().isoformat(),
            }
            raise
        finally:
            self.context.end_time = datetime.now()

        logger.info(f"工作流执行完成: {self.context.state.value}")
        return {
            "execution_id": self.context.execution_id,
            "state": self.context.state.value,
            "results": self.context.node_results,
            "duration": (
                self.context.end_time - self.context.start_time
            ).total_seconds(),
        }

    async def _execute_level(self, node_ids: List[str]):
        """执行一个层级的节点"""
        # 按并行组分组
        parallel_groups = {}
        individual_nodes = []

        for node_id in node_ids:
            node = self.get_node_by_id(node_id)
            if not node:
                continue

            if node.parallel_group:
                if node.parallel_group not in parallel_groups:
                    parallel_groups[node.parallel_group] = []
                parallel_groups[node.parallel_group].append(node)
            else:
                individual_nodes.append(node)

        # 创建所有任务
        tasks = []

        # 添加并行组任务
        for group_name, group_nodes in parallel_groups.items():
            task = self._execute_parallel_group(group_name, group_nodes)
            tasks.append(task)

        # 添加独立节点任务
        for node in individual_nodes:
            task = self._execute_single_node(node)
            tasks.append(task)

        # 并行执行所有任务
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_parallel_group(self, group_name: str, nodes: List[WorkflowNode]):
        """执行并行组"""
        logger.info(f"执行并行组: {group_name}")

        tasks = []
        for node in nodes:
            if self._check_condition(node):
                task = self._execute_single_node(node)
                tasks.append((node.id, task))

        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )
            for (node_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"节点 {node_id} 执行失败: {str(result)}")
                else:
                    self.context.node_results[node_id] = result

    async def _execute_single_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """执行单个节点"""
        logger.info(f"执行节点: {node.name} ({node.id})")
        self.context.current_node = node.id

        # 检查执行条件
        if not self._check_condition(node):
            logger.info(f"跳过节点 {node.name}: 条件不满足")
            return {"skipped": True, "reason": "condition_not_met"}

        try:
            # 创建Agent实例
            agent = await self._create_agent(node)

            # 准备输入数据
            input_data = self._prepare_input_data(node)

            # 执行Agent
            result = await self._execute_with_timeout(agent, input_data, node.timeout)

            logger.info(f"节点 {node.name} 执行成功")
            return result

        except Exception as e:
            logger.error(f"节点 {node.name} 执行失败: {str(e)}")

            # 如果配置了重试，则进行重试
            if node.retry_count > 0:
                for retry in range(node.retry_count):
                    logger.info(f"重试节点 {node.name} (第{retry+1}次)")
                    try:
                        agent = await self._create_agent(node)
                        input_data = self._prepare_input_data(node)
                        result = await self._execute_with_timeout(
                            agent, input_data, node.timeout
                        )
                        logger.info(f"节点 {node.name} 重试成功")
                        return result
                    except Exception as retry_e:
                        logger.error(f"节点 {node.name} 重试失败: {str(retry_e)}")
                        if retry == node.retry_count - 1:  # 最后一次重试
                            raise retry_e
            else:
                raise e

    async def _execute_with_timeout(
        self, agent: BaseAgent, input_data: Dict[str, Any], timeout: Optional[int]
    ) -> Dict[str, Any]:
        """带超时的Agent执行"""
        if timeout:
            return await asyncio.wait_for(
                agent.run_in_workflow(input_data), timeout=timeout
            )
        else:
            return await agent.run_in_workflow(input_data)

    def _check_condition(self, node: WorkflowNode) -> bool:
        """检查节点执行条件"""
        if not node.condition:
            return True

        # 简单的条件表达式解析
        # 支持形如 "node_id.result.key == value" 的表达式
        try:
            # 构建评估上下文
            eval_context = {
                "results": self.context.node_results,
                "global_data": self.context.global_data,
            }

            # 简单的条件评估（生产环境建议使用更安全的表达式解析器）
            return eval(node.condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            logger.warning(f"条件评估失败: {node.condition}, 错误: {str(e)}")
            return False

    async def _create_agent(self, node: WorkflowNode) -> BaseAgent:
        """根据节点配置创建Agent实例"""
        agent_class = self.agent_registry.get(node.agent_type)
        if not agent_class:
            raise ValueError(f"未注册的Agent类型: {node.agent_type}")

        # 合并全局配置和节点配置
        config = {**self.definition.global_config, **node.config}

        # 创建Agent实例
        agent = agent_class(**config)
        agent.set_workflow_context(self.context.global_data, node.id)

        return agent

    def _prepare_input_data(self, node: WorkflowNode) -> Dict[str, Any]:
        """为节点准备输入数据"""
        input_data = {}

        # 从依赖节点获取输出
        for dep_id in node.dependencies:
            if dep_id in self.context.node_results:
                input_data[f"from_{dep_id}"] = self.context.node_results[dep_id]

        # 添加全局数据
        input_data.update(self.context.global_data)

        return input_data

    async def pause(self) -> bool:
        """暂停工作流"""
        if self.context.state == WorkflowState.RUNNING:
            self.context.state = WorkflowState.PAUSED
            logger.info(f"工作流已暂停: {self.definition.name}")
            return True
        return False

    async def resume(self) -> bool:
        """恢复工作流"""
        if self.context.state == WorkflowState.PAUSED:
            self.context.state = WorkflowState.RUNNING
            logger.info(f"工作流已恢复: {self.definition.name}")
            return True
        return False

    async def cancel(self) -> bool:
        """取消工作流"""
        if self.context.state in [WorkflowState.RUNNING, WorkflowState.PAUSED]:
            self.context.state = WorkflowState.CANCELLED
            logger.info(f"工作流已取消: {self.definition.name}")
            return True
        return False
