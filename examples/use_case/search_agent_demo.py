#!/usr/bin/env python3
"""
示例：创建一个使用 ModelSearch 工具的自定义代理
"""

import asyncio
import os
import sys
from typing import List

from pydantic import Field

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.tool import CreateChatCompletion, ModelSearch, Terminate, ToolCollection


class SearchAgent(ToolCallAgent):
    """一个使用模型搜索工具的自定义代理"""

    name: str = "search_agent"
    description: str = "一个能够进行网络搜索并回答问题的代理"

    system_prompt: str = """你是一个强大的搜索助手。你的任务是帮助用户查找信息，并提供有用的回答。

    如果用户询问需要最新信息的问题：
    1. 使用 model_search 工具执行搜索，获取最新信息
    2. 分析搜索结果，提取相关信息
    3. 提供结构化的回答，引用你的信息来源

    如果用户询问一般知识或个人意见：
    1. 使用你自己的知识直接回答
    2. 如果不确定，告诉用户你需要搜索更多信息

    始终保持礼貌和专业，确保你的回答准确、有用且直接。
    """

    next_step_prompt: str = ""  # 不需要每个步骤都询问用户

    available_tools: ToolCollection = ToolCollection(
        ModelSearch(), CreateChatCompletion(), Terminate()
    )

    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 5  # 限制步骤数


async def main():
    """主函数"""
    # 创建搜索代理
    agent = SearchAgent()

    # 测试查询
    queries = [
        "最近的科技发展趋势是什么?",
        "今年中国的经济增长预期如何?",
        "2025年最受欢迎的编程语言是哪些?",
    ]

    # 逐个执行查询
    for query in queries:
        logger.info(f"🔍 处理查询: '{query}'")
        response = await agent.run(query)

        print("\n" + "=" * 50)
        print(f"📝 对于查询 '{query}' 的回答:")
        print("=" * 50)
        print(response)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
