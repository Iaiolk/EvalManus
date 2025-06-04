import asyncio
import time

from app.agent.manus import Manus
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import initialize_logger, logger
from app.prompt.prompt import PromptTemplate

# 确保在主程序入口点初始化日志系统
initialize_logger(name="run_flow")


async def run_flow():
    agents = {
        "manus": Manus(),
    }

    try:
        # prompt_template = open("task/manus准确性任务prompt.txt", "r").read()
        # prompt = PromptTemplate(
        #     input_variables=["query", "session_text"], template=prompt_template
        # )
        # query = "抗日时山东三支队长是谁杨国夫在山东先后什么职务"
        # session_text = "主动检索: 抗日时山东三支队长是谁杨国夫在山东先后什么职务\n———————————————————————————————————————\n| 展现第1位搜索结果, 该结果为AI智能生成\n| AI智能生成资源是利用大模型针对query及相关搜索结果自动生成的内容\n| 用户曝光该资源56.596秒\n———————————————————————————————————————\n——————————————————————————————————————————\n| 展现第2位搜索结果, 该结果为大家都在搜, 此类卡片会推送一批相关query\n——————————————————————————————————————————\n交互点击第1条搜索结果（阿拉丁卡片）\n用户从落地页返回结果列表页\n———————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源30.776秒\n———————————————————————————————\n点击第2位大家都在搜卡片进行激发检索\n用户从落地页返回结果列表页\n——————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源4.294秒\n——————————————————————————————\n"
        # prompt = prompt.format(query=query, session_text=session_text)
        # prompt = "写个hello world的python代码并保存"
        # prompt = input("Enter your prompt: ")
        prompt = """搜索引擎返回结果的准确性一定程度上会影响到用户的使用体验，现在需要你判断搜索引擎中返回的搜索结果中是否有准确性问题。请首先调用有关工具读取task1.txt文件可以获取搜索query和用户搜索行为历史session_text。然后判断搜索query是否属于{事实性问题、数据查询、历史事件}等客观性很强的问题，如果是，则任务该query应该有准确答案，否则则没有准确答案，你可以自行设定详细的判断标准。对于没有准确答案的query，不需要进行后续分析；对于有准确答案的query，你需要接着调用model_search工具获取大模型提供的标准答案，session_text中搜索引擎返回的结果中可能存在多种问题，包括但不限于[前后多个提供结果不一致，与标准答案不一致，时效问题]等多种问题，你可以自行决定具体的判定标准，最后打一个评价分数。把你的判断标准、判断原因和判断结果写入一个结果文件cls_res0.json中。"""

        if prompt.strip().isspace() or not prompt:
            logger.warning("提供了空的提示词。")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("正在处理您的请求...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 整个执行过程60分钟超时
            )
            elapsed_time = time.time() - start_time
            logger.info(f"请求在{elapsed_time:.2f}秒内处理完成")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("请求处理在1小时后超时")
            logger.info("操作因超时而终止。请尝试更简单的请求。")

    except KeyboardInterrupt:
        logger.info("操作被用户取消。")
    except Exception as e:
        logger.error(f"错误：{str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
