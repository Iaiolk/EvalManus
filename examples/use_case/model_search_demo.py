#!/usr/bin/env python3
"""
示例：使用 ModelSearch 工具进行基于模型的网络搜索
这个脚本演示了如何使用 ModelSearch 工具来执行网络搜索
"""

import asyncio
import os
import sys

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.logger import logger
from app.tool import ModelSearch


async def test_model_search(query: str, fetch_content: bool = False):
    """
    测试 ModelSearch 工具

    Args:
        query: 搜索查询
        fetch_content: 是否获取详细内容
    """
    logger.info(f"🔍 开始使用 ModelSearch 搜索: '{query}'")

    # 创建搜索工具实例
    search_tool = ModelSearch()

    # 执行搜索
    response = await search_tool.execute(
        query=query, fetch_content=fetch_content, num_results=3
    )

    # 检查搜索结果
    if response.error:
        logger.error(f"❌ 搜索失败: {response.error}")
        return

    # 打印搜索结果
    print("\n" + "=" * 50)
    print(f"📊 搜索结果: {query}")
    print("=" * 50)
    print(response.output)
    print("=" * 50 + "\n")

    # 返回结果以供进一步处理
    return response


async def main():
    """主函数"""
    queries = ["2024年世界杯", "最新的Python编程教程", "人工智能发展历史"]

    for query in queries:
        await test_model_search(query, fetch_content=True)


if __name__ == "__main__":
    asyncio.run(main())
