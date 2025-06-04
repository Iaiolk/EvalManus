"""
Session评估工作流定义和执行器
"""

from typing import Any, Dict, Type

from app.agent.base import BaseAgent
from app.logger import logger
from app.workflow.base import WorkflowDefinition, WorkflowNode, WorkflowState
from app.workflow.executor import ManusWorkflowExecutor
from app.workflow.session_agents import (
    AccuracyTracingAgent,
    DataPreprocessingAgent,
    FinalScoringAgent,
    IntentRecognitionAgent,
    LLMEvaluationAgent,
    QueryAssessmentAgent,
    SearchAPIAgent,
)


def create_session_eval_workflow() -> WorkflowDefinition:
    """创建Session评估工作流定义"""

    return WorkflowDefinition(
        id="session_evaluation_workflow",
        name="Session行为评估工作流",
        description="对用户session行为进行全面评估和打分的智能工作流",
        version="1.0.0",
        nodes=[
            WorkflowNode(
                id="data_preprocessing",
                name="数据预处理",
                agent_type="data_preprocessing",
                dependencies=[],
                config={
                    "max_steps": 5,
                    "system_prompt": "你是专门处理PV数据和session行为的预处理专家，能够有效清理和转换数据",
                },
                timeout=60,
            ),
            WorkflowNode(
                id="intent_recognition",
                name="用户意图识别",
                agent_type="intent_recognition",
                dependencies=["data_preprocessing"],
                config={
                    "max_steps": 3,
                    "system_prompt": "你是用户意图识别专家，能够准确分析用户行为背后的真实意图",
                },
                timeout=45,
            ),
            WorkflowNode(
                id="llm_evaluation",
                name="LLM评估Session",
                agent_type="llm_evaluation",
                dependencies=["intent_recognition"],
                config={
                    "max_steps": 8,
                    "system_prompt": "你是session评估专家，能够从多个维度全面分析session质量",
                },
                timeout=90,
            ),
            WorkflowNode(
                id="query_assessment",
                name="Query溯源需求评估",
                agent_type="query_assessment",
                dependencies=["llm_evaluation"],
                parallel_group="tracing_prep",
                config={
                    "max_steps": 2,
                    "system_prompt": "你是query分析专家，能够准确判断是否需要进行准确性溯源",
                },
                timeout=30,
            ),
            WorkflowNode(
                id="search_api",
                name="搜索API调用",
                agent_type="search_api",
                dependencies=["query_assessment"],
                condition="results['query_assessment']['_needs_tracing'] == True",
                config={
                    "max_steps": 5,
                    "system_prompt": "你是搜索专家，能够高效调用各种API获取准确信息",
                },
                timeout=120,
                retry_count=2,
            ),
            WorkflowNode(
                id="accuracy_tracing",
                name="准确性溯源",
                agent_type="accuracy_tracing",
                dependencies=["search_api", "llm_evaluation"],
                config={
                    "max_steps": 6,
                    "system_prompt": "你是准确性分析专家，能够基于证据进行深入的溯源分析",
                },
                timeout=75,
            ),
            WorkflowNode(
                id="final_scoring",
                name="最终打分",
                agent_type="final_scoring",
                dependencies=["accuracy_tracing"],
                config={
                    "max_steps": 3,
                    "system_prompt": "你是评分专家，能够综合所有信息给出公正准确的最终分数",
                },
                timeout=45,
            ),
        ],
        global_config={
            "evaluation_criteria": {
                "accuracy": 0.4,
                "relevance": 0.3,
                "completeness": 0.2,
                "efficiency": 0.1,
            },
            "quality_threshold": 0.75,
            "tracing_threshold": 0.85,
        },
        metadata={
            "created_by": "manus_system",
            "category": "session_evaluation",
            "tags": ["evaluation", "session", "accuracy", "llm"],
        },
    )


class SessionEvalWorkflowExecutor(ManusWorkflowExecutor):
    """Session评估工作流执行器"""

    # 注册所有Session评估相关的Agent
    agent_registry: Dict[str, Type[BaseAgent]] = {
        "data_preprocessing": DataPreprocessingAgent,
        "intent_recognition": IntentRecognitionAgent,
        "llm_evaluation": LLMEvaluationAgent,
        "query_assessment": QueryAssessmentAgent,
        "search_api": SearchAPIAgent,
        "accuracy_tracing": AccuracyTracingAgent,
        "final_scoring": FinalScoringAgent,
    }

    def _check_condition(self, node: WorkflowNode) -> bool:
        """重写条件检查，支持Session评估特定的条件逻辑"""
        if not node.condition:
            return True

        try:
            # 支持更复杂的条件表达式
            if "needs_tracing == True" in node.condition:
                query_result = self.context.node_results.get("query_assessment", {})
                return query_result.get("_needs_tracing", False)

            # 支持评分阈值条件
            if "accuracy_score <" in node.condition:
                llm_result = self.context.node_results.get("llm_evaluation", {})
                session_score = llm_result.get("_session_score", {})
                accuracy = session_score.get("accuracy", 1.0)
                threshold = float(node.condition.split("<")[1].strip())
                return accuracy < threshold

            # 其他条件使用父类的通用逻辑
            return super()._check_condition(node)

        except Exception as e:
            logger.warning(f"Session评估条件检查失败: {node.condition}, 错误: {str(e)}")
            return False

    async def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if self.context.state != WorkflowState.COMPLETED:
            return {"status": "incomplete", "message": "工作流尚未完成"}

        final_result = self.context.node_results.get("final_scoring", {})

        return {
            "workflow_id": self.context.workflow_id,
            "execution_id": self.context.execution_id,
            "status": "completed",
            "final_score": final_result.get("_final_score", 0),
            "score_breakdown": final_result.get("_score_breakdown", {}),
            "evaluation_summary": final_result.get("_evaluation_summary", ""),
            "execution_time": (
                (self.context.end_time - self.context.start_time).total_seconds()
                if self.context.end_time
                else None
            ),
            "nodes_executed": len(self.context.node_results),
            "all_results": self.context.node_results,
        }
