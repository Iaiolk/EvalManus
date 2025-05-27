# tool/planning.py
from typing import Dict, List, Literal, Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult

_PLANNING_TOOL_DESCRIPTION = """
允许代理创建和管理计划以解决复杂任务的规划工具。
该工具提供创建计划、更新计划步骤和跟踪进度的功能。
"""


class PlanningTool(BaseTool):
    """
    允许代理创建和管理计划以解决复杂任务的规划工具。
    该工具提供创建计划、更新计划步骤和跟踪进度的功能。
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可用命令：create、update、list、get、set_active、mark_step、delete。",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "计划的唯一标识符。create、update、set_active和delete命令必需。get和mark_step命令可选（如果未指定则使用活动计划）。",
                "type": "string",
            },
            "title": {
                "description": "计划的标题。create命令必需，update命令可选。",
                "type": "string",
            },
            "steps": {
                "description": "计划步骤列表。create命令必需，update命令可选。",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "要更新的步骤索引（从0开始）。mark_step命令必需。",
                "type": "integer",
            },
            "step_status": {
                "description": "为步骤设置的状态。与mark_step命令一起使用。",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "步骤的附加说明。mark_step命令可选。",
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    plans: dict = {}  # 按plan_id存储计划的字典
    _current_plan_id: Optional[str] = None  # 跟踪当前活动计划

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        使用给定的命令和参数执行规划工具。

        参数：
        - command: 要执行的操作
        - plan_id: 计划的唯一标识符
        - title: 计划的标题（与create命令一起使用）
        - steps: 计划的步骤列表（与create命令一起使用）
        - step_index: 要更新的步骤索引（与mark_step命令一起使用）
        - step_status: 为步骤设置的状态（与mark_step命令一起使用）
        - step_notes: 步骤的附加说明（与mark_step命令一起使用）
        """

        if command == "create":
            return self._create_plan(plan_id, title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise ToolError(
                f"无法识别的命令：{command}。允许的命令有：create、update、list、get、set_active、mark_step、delete"
            )

    def _create_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """使用给定的ID、标题和步骤创建新计划。"""
        if not plan_id:
            raise ToolError("命令create需要参数`plan_id`")

        if plan_id in self.plans:
            raise ToolError(
                f"ID为'{plan_id}'的计划已存在。使用'update'来修改现有计划。"
            )

        if not title:
            raise ToolError("命令create需要参数`title`")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError("命令create的参数`steps`必须是非空的字符串列表")

        # 创建一个带有初始化步骤状态的新计划
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # 设置为活跃计划

        return ToolResult(
            output=f"计划创建成功，ID：{plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """使用新标题或步骤更新现有计划。"""
        if not plan_id:
            raise ToolError("命令update需要参数`plan_id`")

        if plan_id not in self.plans:
            raise ToolError(f"未找到ID为：{plan_id}的计划")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError("参数`steps`必须是字符串列表，用于命令：update")

            # 为未更改的步骤保留现有的步骤状态
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # 创建新的步骤状态和说明
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # 如果步骤在旧步骤的相同位置存在，保留状态和说明
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return ToolResult(
            output=f"计划更新成功：{plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """列出所有可用的计划。"""
        if not self.plans:
            return ToolResult(output="没有可用的计划。请使用'create'命令创建计划。")

        output = "可用计划：\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (活跃)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} 步骤已完成"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """获取特定计划的详细信息。"""
        if not plan_id:
            # 如果没有提供plan_id，使用当前活跃计划
            if not self._current_plan_id:
                raise ToolError("没有活跃计划。请指定一个plan_id或设置一个活跃计划。")
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"未找到ID为：{plan_id}的计划")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """将计划设置为活跃计划。"""
        if not plan_id:
            raise ToolError("命令set_active需要参数`plan_id`")

        if plan_id not in self.plans:
            raise ToolError(f"未找到ID为：{plan_id}的计划")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"计划'{plan_id}'现在是活跃计划。\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """标记步骤的特定状态和可选说明。"""
        if not plan_id:
            # 如果没有提供plan_id，使用当前活跃计划
            if not self._current_plan_id:
                raise ToolError("没有活跃计划。请指定一个plan_id或设置一个活跃计划。")
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"未找到ID为：{plan_id}的计划")

        if step_index is None:
            raise ToolError("命令mark_step需要参数`step_index`")

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"无效的step_index：{step_index}。有效索引范围为0到{len(plan['steps'])-1}。"
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"无效的step_status：{step_status}。有效状态为：not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return ToolResult(
            output=f"计划'{plan_id}'中的步骤{step_index}已更新。\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """删除计划。"""
        if not plan_id:
            raise ToolError("命令delete需要参数`plan_id`")

        if plan_id not in self.plans:
            raise ToolError(f"未找到ID为：{plan_id}的计划")

        del self.plans[plan_id]

        # 如果删除的计划是活跃计划，清除活跃计划
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"计划'{plan_id}'已被删除。")

    def _format_plan(self, plan: Dict) -> str:
        """格式化计划以供显示。"""
        output = f"计划：{plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # 计算进度统计
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"进度：{completed}/{total_steps} 步骤已完成 "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"状态：{completed} 已完成，{in_progress} 进行中，{blocked} 被阻塞，{not_started} 未开始\n\n"
        output += "步骤：\n"

        # 添加每个步骤及其状态和说明
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   说明：{notes}\n"

        return output
