import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.search.base import SearchItem


class ModelSearchResult(SearchItem):
    """表示模型搜索返回的单个搜索结果。"""

    source: str = Field(description="搜索结果的来源")
    raw_content: Optional[str] = Field(
        default=None, description="来自搜索结果的原始内容（如果可用）"
    )


class SearchMetadata(ToolResult):
    """有关搜索操作的元数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="找到的结果总数")
    language: str = Field(description="用于搜索的语言代码")
    country: str = Field(description="用于搜索的国家/地区代码")


class ModelSearchResponse(ToolResult):
    """模型搜索工具的结构化响应，继承自ToolResult。"""

    query: str = Field(description="执行的搜索查询")
    results: List[ModelSearchResult] = Field(
        default_factory=list, description="搜索结果列表"
    )
    metadata: Optional[SearchMetadata] = Field(
        default=None, description="有关搜索的元数据"
    )

    @model_validator(mode="after")
    def populate_output(self) -> "ModelSearchResponse":
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
            if result.description and result.description.strip():
                result_text.append(f"   描述: {result.description}")

            # 添加来源信息
            result_text.append(f"   来源: {result.source}")

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


class ModelSearch(BaseTool):
    """使用deepseek-v3的LLM搜索功能搜索网络（默认采用中文搜索模式）。"""

    name: str = "model_search"
    description: str = """使用deepseek-v3的内置网络搜索功能搜索实时信息，默认返回中文结果。
    此工具返回包含相关信息、URL、标题和描述的搜索结果。
    搜索由模型的集成网络搜索功能直接执行，默认采用中文搜索和回答模式。"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(必填) 要提交的搜索查询。",
            },
            "num_results": {
                "type": "integer",
                "description": "(可选) 要返回的最大搜索结果数。默认值为5。",
                "default": 5,
            },
            "lang": {
                "type": "string",
                "description": "(可选) 搜索结果的语言代码（默认值：zh，表示中文）。",
                "default": "zh",
            },
            "country": {
                "type": "string",
                "description": "(可选) 搜索结果的国家/地区代码（默认值：cn，表示中国）。",
                "default": "cn",
            },
            "fetch_content": {
                "type": "boolean",
                "description": "(可选) 是否获取更详细的内容。默认值为false。",
                "default": False,
            },
        },
        "required": ["query"],
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 千帆API配置
    _base_url: str = "https://qianfan.baidubce.com/v2"
    _api_key: str = (
        "bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"
    )
    _model: str = "deepseek-v3"
    client: Any = None

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
        lang: Optional[str] = "zh",  # 默认使用中文
        country: Optional[str] = "cn",  # 默认使用中国地区
        fetch_content: bool = False,
    ) -> ModelSearchResponse:
        """
        使用deepseek-v3的内置网络搜索功能执行搜索。

        Args:
            query: 要提交的搜索查询
            num_results: 要返回的最大搜索结果数（默认值：5）
            lang: 搜索结果的语言代码（默认值："zh"，中文）
            country: 搜索结果的国家/地区代码（默认值："cn"，中国）
            fetch_content: 是否获取更详细的内容（默认值：False）

        Returns:
            包含搜索结果和元数据的结构化响应
        """
        # 如果提供了语言参数，则使用提供的值，否则默认使用中文
        if lang is None:
            lang = (
                getattr(config.search_config, "lang", "zh")
                if config.search_config
                else "zh"
            )

        if country is None:
            country = (
                getattr(config.search_config, "country", "cn")
                if config.search_config
                else "cn"
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

        # 直接使用原始查询，不构建复杂的提示词
        # 执行搜索，最多重试指定次数
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"🔍 执行模型搜索：'{query}'（尝试 {attempt + 1}/{max_retries}）"
                )

                # 调用deepseek-v3的搜索功能
                response_data = await self._perform_model_search(query)

                # 解析搜索结果
                search_results = self._parse_search_results(
                    response_data, query, num_results, fetch_content
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

    async def _perform_model_search(self, query: str) -> dict:
        """
        使用深度搜索模型执行搜索请求。

        Args:
            query: 搜索查询

        Returns:
            包含模型响应内容和搜索结果的字典
        """

        # 构建提示词，明确要求中文结果和回答
        enhanced_prompt = f"""{query}"""
        # 使用异步运行同步API调用
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self._model,
            messages=[{"content": enhanced_prompt, "role": "user"}],
            temperature=0.8,  # 使用较低温度以获得更确定性的输出
            top_p=0.8,
            extra_body={
                "penalty_score": 1.5,
                "web_search": {"enable": True, "enable_trace": True},
                "response_language": "zh",  # 明确指定返回中文结果
            },
        )

        # 返回完整响应对象，包含内容和搜索结果
        result = {"content": "", "search_results": []}

        if response:
            # 提取响应内容
            if response.choices and len(response.choices) > 0:
                result["content"] = response.choices[0].message.content

            # 提取搜索结果
            result["search_results"] = self._extract_search_results_from_response(
                response
            )

        return result

    def _extract_search_results_from_response(self, response: Any) -> List[dict]:
        """
        从API响应中提取搜索结果。

        Args:
            response: API响应对象

        Returns:
            搜索结果列表
        """
        search_results = []

        # 尝试从response.search_results获取
        if hasattr(response, "search_results") and response.search_results:
            search_results = response.search_results
            logger.info(
                f"从response.search_results提取到 {len(search_results)} 个搜索结果"
            )

        # 如果response本身就是一个字典并且包含search_results
        elif isinstance(response, dict) and "search_results" in response:
            search_results = response["search_results"]
            logger.info(
                f"从response[search_results]提取到 {len(search_results)} 个搜索结果"
            )

        # 尝试从response.search_info或其他属性中提取
        elif hasattr(response, "search_info") and hasattr(
            response.search_info, "results"
        ):
            search_results = response.search_info.results
            logger.info(
                f"从response.search_info.results提取到 {len(search_results)} 个搜索结果"
            )

        # 尝试从响应内容中提取搜索引用URL
        elif hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            # 提取可能包含的URL引用
            urls = self._extract_urls_from_text(content)
            if urls:
                logger.info(f"从响应内容中提取到 {len(urls)} 个URL")
                for i, url in enumerate(urls):
                    search_results.append(
                        {"title": f"搜索结果 {i+1}", "url": url, "index": i + 1}
                    )

        # 记录提取结果
        if search_results:
            logger.info(f"总共从API响应中提取到 {len(search_results)} 个搜索结果")
        else:
            logger.warning("未能从API响应中提取到任何搜索结果")

        return search_results

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        从文本中提取URL。

        Args:
            text: 可能包含URL的文本

        Returns:
            提取的URL列表
        """
        if not text:
            return []

        import re

        # URL正则表达式模式，支持中文域名
        url_pattern = r'https?://[^\s<>"\'()]+|www\.[^\s<>"\'()]+'

        # 查找所有匹配
        urls = re.findall(url_pattern, text)

        # 对于不以http开头的URL，添加前缀
        processed_urls = []
        for url in urls:
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            processed_urls.append(url)

        return processed_urls

    def _parse_search_results(
        self, response_data: dict, query: str, num_results: int, fetch_content: bool
    ) -> List[ModelSearchResult]:
        """
        解析模型返回的搜索结果。

        Args:
            response_data: 模型返回的数据，包含内容和搜索结果
            query: 原始查询
            num_results: 期望的结果数量
            fetch_content: 是否获取更详细内容

        Returns:
            解析后的搜索结果列表
        """
        results = []
        content = response_data.get("content", "")
        search_results = response_data.get("search_results", [])

        # 首先尝试直接解析API返回的搜索结果
        if search_results and isinstance(search_results, list):
            logger.info(f"从API搜索结果中提取到 {len(search_results)} 个条目")

            # 限制结果数量
            search_results = (
                search_results[:num_results]
                if len(search_results) > num_results
                else search_results
            )

            for i, item in enumerate(search_results):
                if not isinstance(item, dict):
                    continue

                try:
                    # 提取标题、URL和索引
                    title = item.get("title", "")
                    url = item.get("url", "")
                    index = item.get("index", i + 1)

                    # 优化URL处理，确保中文URL正确处理
                    if url and not (
                        url.startswith("http://") or url.startswith("https://")
                    ):
                        url = "http://" + url

                    # 如果URL非空但title为空，从URL生成一个标题
                    if url and not title:
                        # 从URL中提取域名作为标题
                        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
                        if domain_match:
                            title = f"{domain_match.group(1)} 的搜索结果"
                        else:
                            title = f"结果 {index}"

                    # 使用搜索结果的snippet或content作为描述
                    description = item.get("snippet", "") or item.get("content", "")

                    # 如果没有找到描述，使用完整的模型回答作为描述
                    if not description:
                        description = (
                            content[:300] + "..." if len(content) > 300 else content
                        )

                    results.append(
                        ModelSearchResult(
                            title=title or f"结果 {index}",
                            url=url or "",
                            description=description,
                            source="deepseek-v3",
                            raw_content=content if fetch_content else None,
                        )
                    )
                except Exception as e:
                    logger.error(f"解析API搜索结果项时出错: {str(e)}")

            if results:
                return results

        # 如果无法从API获取搜索结果，回退到从内容中提取JSON
        try:
            # 尝试从响应内容中提取JSON
            json_match = self._extract_json_from_text(content)

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
            logger.warning("未能解析结果，尝试从文本创建简单结果")
            results = self._create_simple_result_from_text(content, query)

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
        # 创建单个搜索结果，确保标题和描述都使用中文
        return [
            ModelSearchResult(
                title=f"问题{query}的搜索结果",
                url="",
                description=text[:500] + ("..." if len(text) > 500 else ""),
                source="deepseek-v3（中文搜索）",
                raw_content=text,
            )
        ]


if __name__ == "__main__":

    async def test_search():
        search_tool = ModelSearch()
        response = await search_tool.execute(
            query="抗日时山东三支队长是谁杨国夫在山东先后什么职务",
            fetch_content=True,
            num_results=3,
            # 不需要显式指定lang和country参数，因为默认已经是"zh"和"cn"
        )
        print(response.output)

    asyncio.run(test_search())
