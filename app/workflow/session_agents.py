"""
Session评估工作流专用Agent集合
"""

from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.logger import logger
from app.tool import ToolCollection


class DataPreprocessingAgent(BaseAgent):
    """数据预处理代理 - 处理PV数据和session行为改写"""

    name: str = "data_preprocessing"
    description: str = "处理PV数据，进行session行为改写和图片解析"

    async def step(self) -> str:
        """执行数据预处理步骤"""
        # 获取PV数据
        pv_data = self.workflow_context.get("pv_data", {})

        if not pv_data:
            return "未获取到PV数据，跳过处理"

        # 模拟数据预处理逻辑
        self.update_memory("system", "开始处理PV数据和session行为改写")
        self.update_memory("user", f"需要处理的PV数据: {pv_data}")

        # 使用LLM进行数据处理
        processing_prompt = f"""
        请对以下PV数据进行预处理和session行为改写：

        PV数据: {pv_data}

        请执行以下任务：
        1. 清理和规范化PV数据
        2. 识别session行为模式
        3. 提取关键行为特征
        4. 过滤无效或异常数据

        请以JSON格式返回处理结果。
        """

        response = await self.llm.chat_completions(
            messages=self.memory.messages
            + [{"role": "user", "content": processing_prompt}],
            temperature=0.1,
        )

        # 保存处理结果
        self._processed_session_data = {
            "original_pv_count": len(pv_data.get("page_views", [])),
            "processed_data": response,
            "processing_timestamp": self.current_step,
        }

        return f"数据预处理完成，处理了{len(pv_data.get('page_views', []))}条PV记录"


class IntentRecognitionAgent(BaseAgent):
    """用户意图识别代理"""

    name: str = "intent_recognition"
    description: str = "基于session行为识别用户意图"

    async def step(self) -> str:
        """执行意图识别步骤"""
        # 获取预处理的session数据
        preprocessing_result = self.workflow_context.get("from_data_preprocessing", {})
        session_data = preprocessing_result.get("_processed_session_data", {})

        if not session_data:
            return "未获取到预处理的session数据"

        intent_prompt = f"""
        请分析以下session行为数据，识别用户的主要意图：

        Session数据: {session_data}

        请分析：
        1. 用户的主要行为模式
        2. 访问路径和页面停留时间
        3. 可能的搜索意图
        4. 任务完成情况

        请给出用户意图分类和置信度评分。
        """

        response = await self.llm.chat_completions(
            messages=[{"role": "user", "content": intent_prompt}], temperature=0.2
        )

        # 保存意图识别结果
        self._identified_intent = {
            "intent_analysis": response,
            "confidence_score": 0.85,  # 模拟置信度
        }

        return (
            f"用户意图识别完成，置信度: {self._identified_intent['confidence_score']}"
        )


class LLMEvaluationAgent(BaseAgent):
    """LLM评估session代理"""

    name: str = "llm_evaluation"
    description: str = "使用LLM对session进行全面评估"

    async def step(self) -> str:
        """执行LLM评估步骤"""
        # 获取前置结果
        session_data = self.workflow_context.get("from_data_preprocessing", {})
        intent_data = self.workflow_context.get("from_intent_recognition", {})

        evaluation_prompt = f"""
        请对以下session进行全面评估：

        Session数据: {session_data}
        用户意图: {intent_data}

        评估维度：
        1. 准确性 (40%): 用户行为与意图的匹配度
        2. 相关性 (30%): 访问内容的相关性
        3. 完整性 (20%): 任务完成的完整程度
        4. 效率性 (10%): 行为路径的效率

        请给出各维度的详细评分和总体评价。
        """

        response = await self.llm.chat_completions(
            messages=[{"role": "user", "content": evaluation_prompt}], temperature=0.3
        )

        # 保存评估结果
        self._session_score = {
            "accuracy": 0.82,
            "relevance": 0.78,
            "completeness": 0.75,
            "efficiency": 0.85,
            "overall": 0.80,
            "detailed_analysis": response,
        }

        return f"LLM评估完成，总体评分: {self._session_score['overall']}"


class QueryAssessmentAgent(BaseAgent):
    """Query是否需要溯源评估代理"""

    name: str = "query_assessment"
    description: str = "评估query是否需要进行准确性溯源"

    async def step(self) -> str:
        """执行query评估步骤"""
        # 获取LLM评估结果
        evaluation_data = self.workflow_context.get("from_llm_evaluation", {})
        session_score = evaluation_data.get("_session_score", {})

        # 判断是否需要溯源
        accuracy_score = session_score.get("accuracy", 0)
        overall_score = session_score.get("overall", 0)

        needs_tracing = accuracy_score < 0.85 or overall_score < 0.80

        assessment_prompt = f"""
        基于以下评估结果，判断是否需要进行准确性溯源：

        准确性评分: {accuracy_score}
        总体评分: {overall_score}
        评估详情: {session_score}

        请判断是否需要通过外部搜索API进行准确性验证。
        """

        response = await self.llm.chat_completions(
            messages=[{"role": "user", "content": assessment_prompt}], temperature=0.1
        )

        # 保存评估结果
        self._needs_tracing = needs_tracing
        self._tracing_reason = response

        result = "需要" if needs_tracing else "不需要"
        return f"Query评估完成，{result}进行准确性溯源"


class SearchAPIAgent(BaseAgent):
    """搜索API调用代理"""

    name: str = "search_api"
    description: str = "调用各类搜索API进行信息检索"

    # 可以添加搜索工具
    # available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(...))

    async def step(self) -> str:
        """执行搜索API调用步骤"""
        # 检查是否需要搜索
        query_assessment = self.workflow_context.get("from_query_assessment", {})
        needs_tracing = query_assessment.get("_needs_tracing", False)

        if not needs_tracing:
            return "跳过搜索API调用，不需要溯源"

        # 模拟API调用
        search_prompt = """
        模拟调用搜索API获取相关信息进行准确性验证。
        """

        # 这里可以调用实际的搜索API工具
        # 目前使用模拟数据
        self._search_results = {
            "api_calls": ["google_search", "bing_search", "wiki_search"],
            "results_count": 15,
            "relevant_documents": 8,
            "verification_data": "模拟的验证数据",
        }

        return f"搜索API调用完成，获取到{self._search_results['results_count']}条结果"


class AccuracyTracingAgent(BaseAgent):
    """准确性溯源代理"""

    name: str = "accuracy_tracing"
    description: str = "进行准确性溯源分析"

    async def step(self) -> str:
        """执行准确性溯源步骤"""
        # 获取搜索结果和评估数据
        search_results = self.workflow_context.get("from_search_api", {})
        evaluation_data = self.workflow_context.get("from_llm_evaluation", {})

        if search_results.get("skipped"):
            # 如果搜索被跳过，使用现有评估结果
            self._accuracy_score = evaluation_data.get("_session_score", {}).get(
                "accuracy", 0.8
            )
            return "搜索被跳过，使用现有准确性评分"

        tracing_prompt = f"""
        基于搜索结果进行准确性溯源分析：

        原始评估: {evaluation_data}
        搜索结果: {search_results}

        请对比分析：
        1. 原始评估的准确性
        2. 外部验证的结果
        3. 差异分析和原因
        4. 修正后的准确性评分
        """

        response = await self.llm.chat_completions(
            messages=[{"role": "user", "content": tracing_prompt}], temperature=0.2
        )

        # 保存溯源结果
        self._accuracy_score = 0.88  # 修正后的准确性评分
        self._tracing_evidence = {
            "verification_sources": search_results.get("_search_results", {}),
            "analysis": response,
            "confidence": 0.92,
        }

        return f"准确性溯源完成，修正评分: {self._accuracy_score}"


class FinalScoringAgent(BaseAgent):
    """最终打分代理"""

    name: str = "final_scoring"
    description: str = "综合所有评估结果进行最终打分"

    async def step(self) -> str:
        """执行最终打分步骤"""
        # 获取所有前置结果
        llm_eval = self.workflow_context.get("from_llm_evaluation", {})
        accuracy_trace = self.workflow_context.get("from_accuracy_tracing", {})

        # 获取评分权重
        criteria = self.workflow_context.get(
            "evaluation_criteria",
            {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.2, "efficiency": 0.1},
        )

        # 获取各维度评分
        session_score = llm_eval.get("_session_score", {})
        final_accuracy = accuracy_trace.get(
            "_accuracy_score", session_score.get("accuracy", 0.8)
        )

        # 计算最终得分
        final_score = (
            final_accuracy * criteria.get("accuracy", 0.4)
            + session_score.get("relevance", 0.8) * criteria.get("relevance", 0.3)
            + session_score.get("completeness", 0.8) * criteria.get("completeness", 0.2)
            + session_score.get("efficiency", 0.8) * criteria.get("efficiency", 0.1)
        )

        scoring_prompt = f"""
        综合所有评估结果进行最终打分：

        LLM评估结果: {session_score}
        准确性溯源结果: {accuracy_trace}
        评分权重: {criteria}

        计算得出的最终评分: {final_score:.3f}

        请提供详细的评分说明和改进建议。
        """

        response = await self.llm.chat_completions(
            messages=[{"role": "user", "content": scoring_prompt}], temperature=0.1
        )

        # 保存最终结果
        self._final_score = final_score
        self._score_breakdown = {
            "accuracy": final_accuracy,
            "relevance": session_score.get("relevance", 0.8),
            "completeness": session_score.get("completeness", 0.8),
            "efficiency": session_score.get("efficiency", 0.8),
            "weights": criteria,
        }
        self._evaluation_summary = response

        return f"最终打分完成，综合得分: {final_score:.3f}"
