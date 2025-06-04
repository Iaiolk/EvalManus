"""
工作流基础类定义
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class WorkflowState(Enum):
    """工作流状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowNode(BaseModel):
    """工作流节点定义"""

    id: str = Field(..., description="节点唯一标识")
    name: str = Field(..., description="节点名称")
    agent_type: str = Field(..., description="对应的Agent类型")
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的节点ID列表"
    )
    config: Dict[str, Any] = Field(default_factory=dict, description="节点配置参数")
    condition: Optional[str] = Field(None, description="执行条件表达式")
    parallel_group: Optional[str] = Field(None, description="并行执行组")
    timeout: Optional[int] = Field(None, description="超时时间(秒)")
    retry_count: int = Field(default=0, description="重试次数")

    class Config:
        extra = "allow"


class WorkflowDefinition(BaseModel):
    """工作流定义"""

    id: str = Field(..., description="工作流唯一标识")
    name: str = Field(..., description="工作流名称")
    description: str = Field(..., description="工作流描述")
    version: str = Field(default="1.0.0", description="工作流版本")
    nodes: List[WorkflowNode] = Field(..., description="节点列表")
    global_config: Dict[str, Any] = Field(default_factory=dict, description="全局配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def validate_dependencies(self) -> bool:
        """验证依赖关系是否合法"""
        node_ids = {node.id for node in self.nodes}
        for node in self.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    raise ValueError(f"节点 {node.id} 的依赖 {dep} 不存在")
        return True

    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（按层级返回）"""
        # 拓扑排序实现
        in_degree = {node.id: 0 for node in self.nodes}
        graph = {node.id: [] for node in self.nodes}

        # 构建图和入度
        for node in self.nodes:
            for dep in node.dependencies:
                graph[dep].append(node.id)
                in_degree[node.id] += 1

        # 分层执行
        levels = []
        current_level = [
            node_id for node_id, degree in in_degree.items() if degree == 0
        ]

        while current_level:
            levels.append(current_level[:])
            next_level = []

            for node_id in current_level:
                for neighbor in graph[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)

            current_level = next_level

        return levels


class WorkflowExecutionContext(BaseModel):
    """工作流执行上下文"""

    workflow_id: str
    execution_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    state: WorkflowState = WorkflowState.PENDING
    current_node: Optional[str] = None
    node_results: Dict[str, Any] = Field(default_factory=dict)
    global_data: Dict[str, Any] = Field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None


class BaseWorkflow(BaseModel, ABC):
    """工作流基类"""

    definition: WorkflowDefinition
    context: WorkflowExecutionContext

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """执行工作流"""
        pass

    @abstractmethod
    async def pause(self) -> bool:
        """暂停工作流"""
        pass

    @abstractmethod
    async def resume(self) -> bool:
        """恢复工作流"""
        pass

    @abstractmethod
    async def cancel(self) -> bool:
        """取消工作流"""
        pass

    def get_node_by_id(self, node_id: str) -> Optional[WorkflowNode]:
        """根据ID获取节点"""
        for node in self.definition.nodes:
            if node.id == node_id:
                return node
        return None
