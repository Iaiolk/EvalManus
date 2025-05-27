import asyncio
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.search import (
    BaiduSearchEngine,
    BingSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)
from app.tool.search.base import SearchItem


class SearchResult(BaseModel):
    """表示搜索引擎返回的单个搜索结果。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="搜索结果中的位置")
    url: str = Field(description="搜索结果的URL")
    title: str = Field(default="", description="搜索结果的标题")
    description: str = Field(default="", description="搜索结果的描述或摘要")
    source: str = Field(description="提供此结果的搜索引擎")
    raw_content: Optional[str] = Field(
        default=None, description="搜索结果页面的原始内容（如果可用）"
    )

    def __str__(self) -> str:
        """搜索结果的字符串表示。"""
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    """关于搜索操作的元数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="找到的结果总数")
    language: str = Field(description="搜索使用的语言代码")
    country: str = Field(description="搜索使用的国家代码")


class SearchResponse(ToolResult):
    """网页搜索工具的结构化响应，继承自ToolResult。"""

    query: str = Field(description="执行的搜索查询")
    results: List[SearchResult] = Field(
        default_factory=list, description="搜索结果列表"
    )
    metadata: Optional[SearchMetadata] = Field(
        default=None, description="关于搜索的元数据"
    )

    @model_validator(mode="after")
    def populate_output(self) -> "SearchResponse":
        """根据搜索结果填充输出或错误字段。"""
        if self.error:
            return self

        result_text = [f"'{self.query}'的搜索结果："]

        for i, result in enumerate(self.results, 1):
            # 添加带位置编号的标题
            title = result.title.strip() or "无标题"
            result_text.append(f"\n{i}. {title}")

            # 添加带适当缩进的URL
            result_text.append(f"   网址: {result.url}")

            # 如果有描述则添加
            if result.description.strip():
                result_text.append(f"   描述: {result.description}")

            # 如果有内容预览则添加
            if result.raw_content:
                content_preview = result.raw_content[:1000].replace("\n", " ").strip()
                if len(result.raw_content) > 1000:
                    content_preview += "..."
                result_text.append(f"   内容: {content_preview}")

        # 如果有元数据则在底部添加
        if self.metadata:
            result_text.extend(
                [
                    f"\n元数据:",
                    f"- 总结果数: {self.metadata.total_results}",
                    f"- 语言: {self.metadata.language}",
                    f"- 国家/地区: {self.metadata.country}",
                ]
            )

        self.output = "\n".join(result_text)
        return self


class WebContentFetcher:
    """Utility class for fetching web content."""

    @staticmethod
    async def fetch_content(url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch and extract the main content from a webpage.

        Args:
            url: The URL to fetch content from
            timeout: Request timeout in seconds

        Returns:
            Extracted text content or None if fetching fails
        """
        headers = {
            "WebSearch": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            # Use asyncio to run requests in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=timeout)
            )

            if response.status_code != 200:
                logger.warning(
                    f"Failed to fetch content from {url}: HTTP {response.status_code}"
                )
                return None

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace and limit size (100KB max)
            text = " ".join(text.split())
            return text[:10000] if text else None

        except Exception as e:
            logger.warning(f"Error fetching content from {url}: {e}")
            return None


class WebSearch(BaseTool):
    """Search the web for information using various search engines."""

    name: str = "web_search"
    description: str = """Search the web for real-time information about any topic.
    This tool returns comprehensive search results with relevant information, URLs, titles, and descriptions.
    If the primary search engine fails, it automatically falls back to alternative engines."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 5.",
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
                "description": "(optional) Whether to fetch full content from result pages. Default is false.",
                "default": False,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
        "bing": BingSearchEngine(),
    }
    content_fetcher: WebContentFetcher = WebContentFetcher()

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        lang: Optional[str] = None,
        country: Optional[str] = None,
        fetch_content: bool = False,
    ) -> SearchResponse:
        """
        Execute a Web search and return detailed search results.

        Args:
            query: The search query to submit to the search engine
            num_results: The number of search results to return (default: 5)
            lang: Language code for search results (default from config)
            country: Country code for search results (default from config)
            fetch_content: Whether to fetch content from result pages (default: False)

        Returns:
            A structured response containing search results and metadata
        """
        # Get settings from config
        retry_delay = (
            getattr(config.search_config, "retry_delay", 60)
            if config.search_config
            else 60
        )
        max_retries = (
            getattr(config.search_config, "max_retries", 3)
            if config.search_config
            else 3
        )

        # Use config values for lang and country if not specified
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

        search_params = {"lang": lang, "country": country}

        # Try searching with retries when all engines fail
        for retry_count in range(max_retries + 1):
            results = await self._try_all_engines(query, num_results, search_params)

            if results:
                # Fetch content if requested
                if fetch_content:
                    results = await self._fetch_content_for_results(results)

                # Return a successful structured response
                return SearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=lang,
                        country=country,
                    ),
                )

            if retry_count < max_retries:
                # All engines failed, wait and retry
                logger.warning(
                    f"All search engines failed. Waiting {retry_delay} seconds before retry {retry_count + 1}/{max_retries}..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"All search engines failed after {max_retries} retries. Giving up."
                )

        # Return an error response
        return SearchResponse(
            query=query,
            error="All search engines failed to return results after multiple retries.",
            results=[],
        )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """Try all search engines in the configured order."""
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            logger.info(f"🔎 Attempting search with {engine_name.capitalize()}...")
            search_items = await self._perform_search_with_engine(
                engine, query, num_results, search_params
            )

            if not search_items:
                continue

            if failed_engines:
                logger.info(
                    f"Search successful with {engine_name.capitalize()} after trying: {', '.join(failed_engines)}"
                )

            # Transform search items into structured results
            return [
                SearchResult(
                    position=i + 1,
                    url=item.url,
                    title=item.title
                    or f"Result {i+1}",  # Ensure we always have a title
                    description=item.description or "",
                    source=engine_name,
                )
                for i, item in enumerate(search_items)
            ]

        if failed_engines:
            logger.error(f"All search engines failed: {', '.join(failed_engines)}")
        return []

    async def _fetch_content_for_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fetch and add web content to search results."""
        if not results:
            return []

        # Create tasks for each result
        tasks = [self._fetch_single_result_content(result) for result in results]

        # Type annotation to help type checker
        fetched_results = await asyncio.gather(*tasks)

        # Explicit validation of return type
        return [
            (
                result
                if isinstance(result, SearchResult)
                else SearchResult(**result.dict())
            )
            for result in fetched_results
        ]

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        """Fetch content for a single search result."""
        if result.url:
            content = await self.content_fetcher.fetch_content(result.url)
            if content:
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        """Determines the order in which to try search engines."""
        preferred = (
            getattr(config.search_config, "engine", "google").lower()
            if config.search_config
            else "google"
        )
        fallbacks = (
            [engine.lower() for engine in config.search_config.fallback_engines]
            if config.search_config
            and hasattr(config.search_config, "fallback_engines")
            else []
        )

        # Start with preferred engine, then fallbacks, then remaining engines
        engine_order = [preferred] if preferred in self._search_engine else []
        engine_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self._search_engine and fb not in engine_order
            ]
        )
        engine_order.extend([e for e in self._search_engine if e not in engine_order])

        return engine_order

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Execute search with the given engine and parameters."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(
                engine.perform_search(
                    query,
                    num_results=num_results,
                    lang=search_params.get("lang"),
                    country=search_params.get("country"),
                )
            ),
        )


if __name__ == "__main__":
    web_search = WebSearch()
    search_response = asyncio.run(
        web_search.execute(
            query="Python programming", fetch_content=True, num_results=1
        )
    )
    print(search_response.to_tool_result())
