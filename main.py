import asyncio

from app.agent.manus import Manus
from app.logger import logger
from app.prompt.prompt import PromptTemplate


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        prompt_template = open("task/manus准确性任务prompt.txt", "r").read()
        prompt = PromptTemplate(
            input_variables=["query", "session_text"], template=prompt_template
        )
        query = "抗日时山东三支队长是谁杨国夫在山东先后什么职务"
        session_text = "主动检索: 抗日时山东三支队长是谁杨国夫在山东先后什么职务\n———————————————————————————————————————\n| 展现第1位搜索结果, 该结果为AI智能生成\n| AI智能生成资源是利用大模型针对query及相关搜索结果自动生成的内容\n| 用户曝光该资源56.596秒\n———————————————————————————————————————\n——————————————————————————————————————————\n| 展现第2位搜索结果, 该结果为大家都在搜, 此类卡片会推送一批相关query\n——————————————————————————————————————————\n交互点击第1条搜索结果（阿拉丁卡片）\n用户从落地页返回结果列表页\n———————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源30.776秒\n———————————————————————————————\n点击第2位大家都在搜卡片进行激发检索\n用户从落地页返回结果列表页\n——————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源4.294秒\n——————————————————————————————\n"
        prompt = prompt.format(query=query, session_text=session_text)

        # prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
