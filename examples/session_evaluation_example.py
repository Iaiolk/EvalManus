"""
Session评估工作流使用示例
"""

import asyncio
import json
from datetime import datetime

from app.logger import logger
from app.workflow.session_eval import (
    SessionEvalWorkflowExecutor,
    create_session_eval_workflow,
)


async def run_session_evaluation_example():
    """运行Session评估工作流示例"""

    # 模拟PV数据
    sample_pv_data = {
        "user_id": "user_12345",
        "session_id": "sess_abc123",
        "page_views": [
            {
                "url": "https://example.com/search?q=python+tutorial",
                "timestamp": "2024-01-15T10:00:00Z",
                "duration": 45,
                "actions": ["search", "click"],
            },
            {
                "url": "https://example.com/tutorial/python-basics",
                "timestamp": "2024-01-15T10:00:45Z",
                "duration": 180,
                "actions": ["read", "scroll", "bookmark"],
            },
            {
                "url": "https://example.com/tutorial/python-advanced",
                "timestamp": "2024-01-15T10:03:45Z",
                "duration": 120,
                "actions": ["read", "copy_code"],
            },
        ],
        "user_actions": [
            {
                "type": "search",
                "query": "python tutorial",
                "timestamp": "2024-01-15T10:00:00Z",
            },
            {
                "type": "click",
                "target": "tutorial_link",
                "timestamp": "2024-01-15T10:00:30Z",
            },
            {
                "type": "bookmark",
                "page": "python-basics",
                "timestamp": "2024-01-15T10:02:00Z",
            },
        ],
        "session_metadata": {
            "device": "desktop",
            "browser": "chrome",
            "location": "CN",
            "total_duration": 345,
        },
    }

    try:
        # 创建工作流定义
        workflow_def = create_session_eval_workflow()
        logger.info(f"创建工作流: {workflow_def.name}")

        # 创建执行器
        executor = SessionEvalWorkflowExecutor(workflow_def)

        # 设置初始数据
        executor.context.global_data = {
            "pv_data": sample_pv_data,
            "evaluation_start_time": datetime.now().isoformat(),
        }

        logger.info("开始执行Session评估工作流...")

        # 执行工作流
        execution_result = await executor.execute()

        logger.info("工作流执行完成!")

        # 获取评估摘要
        summary = await executor.get_evaluation_summary()

        # 输出结果
        print("\n" + "=" * 60)
        print("SESSION评估结果摘要")
        print("=" * 60)
        print(f"执行ID: {summary['execution_id']}")
        print(f"状态: {summary['status']}")
        print(f"最终评分: {summary['final_score']:.3f}")
        print(f"执行时间: {summary['execution_time']:.2f}秒")
        print(f"执行节点数: {summary['nodes_executed']}")

        print("\n评分明细:")
        breakdown = summary.get("score_breakdown", {})
        for dimension, score in breakdown.items():
            if dimension != "weights" and isinstance(score, (int, float)):
                print(f"  {dimension}: {score:.3f}")

        print(f"\n评估总结:")
        print(summary.get("evaluation_summary", ""))

        # 保存详细结果到文件
        result_file = (
            f"workspace/session_eval_result_{executor.context.execution_id}.json"
        )
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n详细结果已保存到: {result_file}")

        return summary

    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        raise


async def run_batch_evaluation():
    """批量评估示例"""

    batch_data = [
        {
            "session_id": "sess_001",
            "user_id": "user_001",
            "page_views": [{"url": "example.com/page1", "duration": 60}],
        },
        {
            "session_id": "sess_002",
            "user_id": "user_002",
            "page_views": [{"url": "example.com/page2", "duration": 120}],
        },
    ]

    results = []

    for session_data in batch_data:
        logger.info(f"评估Session: {session_data['session_id']}")

        workflow_def = create_session_eval_workflow()
        executor = SessionEvalWorkflowExecutor(workflow_def)
        executor.context.global_data = {"pv_data": session_data}

        try:
            await executor.execute()
            summary = await executor.get_evaluation_summary()
            results.append(summary)

        except Exception as e:
            logger.error(f"Session {session_data['session_id']} 评估失败: {str(e)}")
            results.append(
                {
                    "session_id": session_data["session_id"],
                    "status": "failed",
                    "error": str(e),
                }
            )

    # 保存批量结果
    batch_result_file = (
        f"workspace/batch_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(batch_result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"批量评估完成，结果保存到: {batch_result_file}")
    return results


if __name__ == "__main__":
    # 运行单个评估示例
    asyncio.run(run_session_evaluation_example())

    # 运行批量评估示例
    # asyncio.run(run_batch_evaluation())
