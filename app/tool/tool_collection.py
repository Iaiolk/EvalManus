"""管理多个工具的集合类。"""

from typing import Any, Dict, List

from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """已定义工具的集合。"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"工具{name}无效")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """按顺序执行集合中的所有工具。"""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """将单个工具添加到集合中。

        如果已存在同名工具，将跳过该工具并记录警告。
        """
        if tool.name in self.tool_map:
            logger.warning(f"工具{tool.name}已存在于集合中，跳过")
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """将多个工具添加到集合中。

        如果任何工具与现有工具有名称冲突，将跳过该工具并记录警告。
        """
        for tool in tools:
            self.add_tool(tool)
        return self
