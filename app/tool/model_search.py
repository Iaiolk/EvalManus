import asyncio
import json
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.search.base import SearchItem


class ModelSearchResult(SearchItem):
    """Represents a single search result returned by the model search."""

    source: str = Field(description="The source of the search result")
    raw_content: Optional[str] = Field(
        default=None, description="Raw content from the search result if available"
    )


class SearchMetadata(ToolResult):
    """Metadata about the search operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="Total number of results found")
    language: str = Field(description="Language code used for the search")
    country: str = Field(description="Country code used for the search")


class ModelSearchResponse(ToolResult):
    """Structured response from the model search tool, inheriting ToolResult."""

    query: str = Field(description="The search query that was executed")
    results: List[ModelSearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    metadata: Optional[SearchMetadata] = Field(
        default=None, description="Metadata about the search"
    )

    @model_validator(mode="after")
    def populate_output(self) -> "ModelSearchResponse":
        """Populate output or error fields based on search results."""
        if self.error:
            return self

        result_text = [f"Search results for '{self.query}':"]

        for i, result in enumerate(self.results, 1):
            # Add title with position number
            title = result.title.strip() or "No title"
            result_text.append(f"\n{i}. {title}")

            # Add URL with proper indentation
            result_text.append(f"   URL: {result.url}")

            # Add description if available
            if result.description and result.description.strip():
                result_text.append(f"   Description: {result.description}")

            # Add source info
            result_text.append(f"   Source: {result.source}")

            # Add content preview if available
            if result.raw_content:
                content_preview = result.raw_content[:1000].replace("\n", " ").strip()
                if len(result.raw_content) > 1000:
                    content_preview += "..."
                result_text.append(f"   Content: {content_preview}")

        # Add metadata at the bottom if available
        if self.metadata:
            result_text.extend(
                [
                    f"\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )

        self.output = "\n".join(result_text)
        return self


class ModelSearch(BaseTool):
    """Search the web using LLM-powered search capabilities via deepseek-v3."""

    name: str = "model_search"
    description: str = """Search the web for real-time information using deepseek-v3's built-in web search.
    This tool returns search results with relevant information, URLs, titles, and descriptions.
    The search is performed directly by the model's integrated web search capability."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The maximum number of search results to return. Default is 5.",
                "default": 5,
            },
            "lang": {
                "type": "string",
                "description": "(optional) Language code for search results (default: en).",
                "default": "en",
            },
            "country": {
                "type": "string",
                "description": "(optional) Country code for search results (default: us).",
                "default": "us",
            },
            "fetch_content": {
                "type": "boolean",
                "description": "(optional) Whether to fetch more detailed content. Default is false.",
                "default": False,
            },
        },
        "required": ["query"],
    }

    # 千帆API配置
    _base_url: str = "https://qianfan.baidubce.com/v2"
    _api_key: str = (
        "bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"
    )
    _model: str = "deepseek-v3"

    # 初始化OpenAI客户端
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        lang: Optional[str] = None,
        country: Optional[str] = None,
        fetch_content: bool = False,
    ) -> ModelSearchResponse:
        """
        Execute a search using deepseek-v3's built-in web search capability.

        Args:
            query: The search query to submit
            num_results: The maximum number of search results to return (default: 5)
            lang: Language code for search results (default from config)
            country: Country code for search results (default from config)
            fetch_content: Whether to fetch more detailed content (default: False)

        Returns:
            A structured response containing search results and metadata
        """
        # 使用配置值设置语言和国家代码
        if lang is None:
            lang = (
                getattr(config.search_config, "lang", "en")
                if config.search_config
                else "en"
            )

        if country is None:
            country = (
                getattr(config.search_config, "country", "us")
                if config.search_config
                else "us"
            )

        # 获取重试配置
        max_retries = (
            getattr(config.search_config, "max_retries", 3)
            if config.search_config
            else 3
        )
        retry_delay = (
            getattr(config.search_config, "retry_delay", 5)
            if config.search_config
            else 5
        )

        # 准备搜索提示词
        if fetch_content:
            search_prompt = f"""请使用网络搜索查找关于"{query}"的信息。
            请返回至少{num_results}个相关结果，包括标题、URL、描述和内容摘要。
            使用JSON格式返回，包括以下字段：title, url, description, content。
            确保结果是最新的，并尽可能提供详细内容。"""
        else:
            search_prompt = f"""请使用网络搜索查找关于"{query}"的信息。
            请返回至少{num_results}个相关结果，包括标题、URL和简短描述。
            使用JSON格式返回，包括以下字段：title, url, description。
            确保结果是最新的。"""

        # 添加语言和国家提示
        search_prompt += (
            f"\n请优先提供{lang}语言的结果，尤其是来自{country}地区的信息。"
        )

        # 执行搜索，最多重试指定次数
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"🔍 执行模型搜索：'{query}'（尝试 {attempt + 1}/{max_retries}）"
                )

                # 调用deepseek-v3的搜索功能
                response = await self._perform_model_search(search_prompt)

                # 解析搜索结果
                search_results = self._parse_search_results(
                    response, query, num_results, fetch_content
                )

                if search_results and len(search_results) > 0:
                    logger.info(f"✅ 成功获取到 {len(search_results)} 个搜索结果")

                    # 返回成功的响应
                    return ModelSearchResponse(
                        status="success",
                        query=query,
                        results=search_results,
                        metadata=SearchMetadata(
                            total_results=len(search_results),
                            language=lang,
                            country=country,
                        ),
                    )
            except Exception as e:
                logger.error(f"🚨 模型搜索失败: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)

        # 所有尝试都失败后返回错误响应
        return ModelSearchResponse(
            query=query,
            error="无法获取搜索结果，所有尝试均失败。",
            results=[],
        )

    async def _perform_model_search(self, query: str) -> str:
        """
        使用深度搜索模型执行搜索请求。

        Args:
            query: 搜索查询

        Returns:
            模型的响应内容
        """
        # 使用异步运行同步API调用
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self._model,
            messages=[{"content": query, "role": "user"}],
            temperature=0.2,  # 使用较低温度以获得更确定性的输出
            top_p=0.8,
            extra_body={
                "penalty_score": 1,
                "web_search": {"enable": True, "enable_trace": True},
            },
        )

        # 提取响应内容
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return ""

    def _parse_search_results(
        self, response_text: str, query: str, num_results: int, fetch_content: bool
    ) -> List[ModelSearchResult]:
        """
        解析模型返回的搜索结果。

        Args:
            response_text: 模型返回的文本
            query: 原始查询
            num_results: 期望的结果数量
            fetch_content: 是否获取更详细内容

        Returns:
            解析后的搜索结果列表
        """
        results = []

        try:
            # 尝试从响应中提取JSON
            json_match = self._extract_json_from_text(response_text)

            if json_match:
                # 解析JSON结果
                json_data = json.loads(json_match)

                # 处理可能的不同JSON结构
                items = []
                if isinstance(json_data, list):
                    items = json_data
                elif isinstance(json_data, dict) and "results" in json_data:
                    items = json_data["results"]
                elif isinstance(json_data, dict) and "items" in json_data:
                    items = json_data["items"]

                # 限制结果数量
                items = items[:num_results] if items else []

                # 创建ModelSearchResult对象
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue

                    try:
                        results.append(
                            ModelSearchResult(
                                title=item.get("title", f"Result {i+1}"),
                                url=item.get("url", ""),
                                description=item.get("description", ""),
                                source="deepseek-v3",
                                raw_content=(
                                    item.get("content", "") if fetch_content else None
                                ),
                            )
                        )
                    except Exception as e:
                        logger.error(f"解析搜索结果项时出错: {str(e)}")
        except Exception as e:
            logger.error(f"解析搜索结果时出错: {str(e)}")

        # 如果没有提取到结果，尝试从文本中创建简单结果
        if not results:
            logger.warning("未能解析JSON结果，尝试从文本创建简单结果")
            results = self._create_simple_result_from_text(response_text, query)

        return results

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        从文本中提取JSON字符串。

        Args:
            text: 可能包含JSON的文本

        Returns:
            提取的JSON字符串，如果没有找到则返回None
        """
        # 尝试找到JSON代码块
        import re

        json_pattern = r"```(?:json)?\n([\s\S]*?)\n```"
        json_matches = re.findall(json_pattern, text)

        if json_matches:
            return json_matches[0]

        # 尝试找到大括号包围的整个文本
        if text.strip().startswith("{") and text.strip().endswith("}"):
            return text

        # 尝试找到方括号包围的整个文本
        if text.strip().startswith("[") and text.strip().endswith("]"):
            return text

        return None

    def _create_simple_result_from_text(
        self, text: str, query: str
    ) -> List[ModelSearchResult]:
        """
        当JSON解析失败时，尝试从文本中创建简单搜索结果。

        Args:
            text: 模型返回的文本
            query: 原始查询

        Returns:
            创建的搜索结果列表
        """
        # 创建单个搜索结果
        return [
            ModelSearchResult(
                title=f"搜索结果: {query}",
                url="",
                description=text[:500] + ("..." if len(text) > 500 else ""),
                source="deepseek-v3",
                raw_content=text,
            )
        ]


if __name__ == "__main__":

    async def test_search():
        search_tool = ModelSearch()
        response = await search_tool.execute(
            query="Python编程入门教程", fetch_content=True, num_results=3
        )
        print(response.output)

    asyncio.run(test_search())
