import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.tool.model_search import ModelSearch, ModelSearchResponse, ModelSearchResult


class TestModelSearch(unittest.TestCase):
    """测试ModelSearch工具类的单元测试。"""

    def setUp(self):
        """测试前的准备工作。"""
        # 创建ModelSearch实例，但使用模拟替代真正的OpenAI客户端
        self.model_search = ModelSearch()
        self.model_search.client = MagicMock()

        # 设置一些测试数据
        self.test_query = "抗日时山东三支队长是谁杨国夫在山东先后什么职务"
        self.mock_response_content = """
        抗日战争(1937-1945)是中国人民抵抗日本侵略的伟大战争。主要历史事件包括：
        1. 卢沟桥事变(1937年7月7日)：抗日战争全面爆发的标志
        2. 平型关大捷(1937年9月)：八路军首次大捷
        3. 台儿庄战役(1938年)：国民党军队重要胜利
        4. 百团大战(1940年)：八路军对日军发动的规模最大战役

        有关山东抗日武装领导人，杨国夫曾任山东抗日游击第三支队司令员，后来担任山东军区第三军分区司令员等职务。
        """

    def test_create_simple_result_from_text(self):
        """测试从文本创建简单搜索结果的功能。"""
        # 调用测试方法
        results = self.model_search._create_simple_result_from_text(
            self.mock_response_content, self.test_query
        )

        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "deepseek-v3（中文搜索）")
        self.assertEqual(results[0].title, f"问题{self.test_query}的搜索结果")
        # 检查描述是否包含原始内容的开头部分
        self.assertIn(self.mock_response_content[:10], results[0].description)
        self.assertEqual(results[0].raw_content, self.mock_response_content)

    def test_extract_urls_from_text(self):
        """测试从文本中提取URL的功能。"""
        test_text = "这是一个包含URL的文本 https://www.example.com 和 http://test.cn/page?q=123 以及 www.baidu.com"
        urls = self.model_search._extract_urls_from_text(test_text)

        # 验证提取结果
        self.assertEqual(len(urls), 3)
        self.assertIn("https://www.example.com", urls)
        self.assertIn("http://test.cn/page?q=123", urls)
        self.assertIn("http://www.baidu.com", urls)

    def test_perform_model_search(self):
        """测试执行模型搜索的功能。"""

        async def _async_test():
            # 创建模拟的API响应
            mock_api_response = MagicMock()
            mock_api_response.choices = [MagicMock()]
            mock_api_response.choices[0].message.content = self.mock_response_content

            # 设置异步模拟
            with patch(
                "app.tool.model_search.asyncio.to_thread",
                return_value=mock_api_response,
            ):
                # 执行被测试方法
                result = await self.model_search._perform_model_search(self.test_query)

                # 验证结果
                self.assertEqual(result["content"], self.mock_response_content)

                # 验证API调用
                # 验证是否调用了chat.completions.create方法（此处简化验证）
                self.assertTrue("web_search" in str(asyncio.to_thread.call_args))

        # 运行异步测试
        asyncio.run(_async_test())

    def test_execute_success(self):
        """测试成功执行搜索的场景。"""

        async def _async_test():
            # 设置模拟响应
            mock_search_result = ModelSearchResult(
                title="测试结果",
                url="https://example.com",
                description="测试描述",
                source="deepseek-v3",
                raw_content=self.mock_response_content,
            )

            # 使用上下文管理器模拟方法
            with patch.object(
                ModelSearch,
                "_perform_model_search",
                return_value={
                    "content": self.mock_response_content,
                    "search_results": [],
                },
            ) as mock_perform, patch.object(
                ModelSearch, "_parse_search_results", return_value=[mock_search_result]
            ) as mock_parse:
                # 执行测试
                response = await self.model_search.execute(query=self.test_query)

                # 验证结果
                # 使用error属性检查是否成功（error为None表示成功）
                self.assertIsNone(response.error)
                self.assertEqual(response.query, self.test_query)
                self.assertEqual(len(response.results), 1)
                self.assertEqual(response.results[0].title, "测试结果")
                self.assertEqual(response.metadata.language, "zh")
                self.assertEqual(response.metadata.country, "cn")

                # 验证方法调用
                mock_perform.assert_called_once_with(self.test_query)
                mock_parse.assert_called_once()

        # 运行异步测试
        asyncio.run(_async_test())

    def test_execute_error(self):
        """测试执行搜索失败的场景。"""

        async def _async_test():
            # 使用上下文管理器模拟方法抛出异常
            with patch.object(
                ModelSearch, "_perform_model_search", side_effect=Exception("API错误")
            ):
                # 执行测试
                response = await self.model_search.execute(query=self.test_query)

                # 验证结果
                self.assertIsNotNone(response.error)
                self.assertEqual(response.results, [])

        # 运行异步测试
        asyncio.run(_async_test())

    def test_parse_search_results_with_api_results(self):
        """测试解析API返回的搜索结果。"""
        # 创建测试数据
        api_response = {
            "content": "测试内容",
            "search_results": [
                {"title": "结果1", "url": "https://example.com/1", "snippet": "描述1"},
                {"title": "结果2", "url": "https://example.com/2", "snippet": "描述2"},
            ],
        }

        # 调用测试方法
        results = self.model_search._parse_search_results(
            api_response, self.test_query, 2, False
        )

        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "结果1")
        self.assertEqual(results[0].url, "https://example.com/1")
        self.assertEqual(results[0].description, "描述1")
        self.assertIsNone(results[0].raw_content)  # fetch_content为False

    def test_parse_search_results_with_json_content(self):
        """测试从JSON内容中解析搜索结果。"""
        # 创建带有JSON内容的响应
        json_content = '{"results": [{"title": "JSON结果", "url": "https://json.example.com", "description": "JSON描述"}]}'
        api_response = {
            "content": f"```json\n{json_content}\n```",
            "search_results": [],  # 空搜索结果，强制从内容中提取JSON
        }

        # 模拟方法以返回JSON字符串
        with patch.object(
            self.model_search, "_extract_json_from_text", return_value=json_content
        ):
            # 调用测试方法
            results = self.model_search._parse_search_results(
                api_response, self.test_query, 2, False
            )

            # 验证结果
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "JSON结果")
            self.assertEqual(results[0].url, "https://json.example.com")

    def test_extract_json_from_text(self):
        """测试从文本中提取JSON的功能。"""
        # 测试代码块中的JSON
        code_block_text = (
            '以下是搜索结果：\n```json\n{"results": [{"title": "测试"}]}\n```\n更多信息'
        )
        json_str = self.model_search._extract_json_from_text(code_block_text)
        self.assertEqual(json_str, '{"results": [{"title": "测试"}]}')

        # 测试整个文本是JSON
        json_text = '{"key": "value"}'
        json_str = self.model_search._extract_json_from_text(json_text)
        self.assertEqual(json_str, json_text)


def run_async_test(coroutine):
    """执行异步测试的辅助函数。"""
    return asyncio.run(coroutine)


if __name__ == "__main__":
    unittest.main()
