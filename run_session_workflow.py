#!/usr/bin/env python3
"""
Manus Session评估工作流运行脚本
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from app.logger import logger
from examples.session_evaluation_example import (
    run_batch_evaluation,
    run_session_evaluation_example,
)


def main():
    parser = argparse.ArgumentParser(description="运行Manus Session评估工作流")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="运行模式: single (单个评估) 或 batch (批量评估)",
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.mode == "single":
            logger.info("运行单个Session评估示例...")
            result = asyncio.run(run_session_evaluation_example())
            print(f"\n✅ 单个评估完成，最终评分: {result.get('final_score', 0):.3f}")

        elif args.mode == "batch":
            logger.info("运行批量Session评估示例...")
            results = asyncio.run(run_batch_evaluation())
            print(f"\n✅ 批量评估完成，处理了 {len(results)} 个Session")

            # 统计成功率
            success_count = sum(1 for r in results if r.get("status") == "completed")
            print(
                f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)"
            )

    except KeyboardInterrupt:
        print("\n❌ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        print(f"\n❌ 执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
