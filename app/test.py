from typing import List


def format_ernie_messages(messages: List[dict]) -> List[dict]:
    """
    按照百度文心一言API的规则格式化消息列表。

    规则：
    1. 第一条message的role必须是user或system
    2. 最后一条message的role必须是user或tool (对于ERNIE 4.5或X1 Turbo系列,必须是user)
    3. 消息顺序规则:
        - 未使用function call: user->assistant->user...交替
        - 使用function call时:
            - 第一条是user: user->assistant->user/tool...
            - 第一条是system: system->user/tool->assistant->user/tool...
    4. 连续的两条user消息将被合并，第一条前添加"用户输入："，第二条前添加"\n执行要求:\n"
    5. 如果一条消息的role为tool，它后面的一条消息role为user，则删除这条user消息

    Args:
        messages: 原始消息列表

    Returns:
        List[dict]: 格式化后的消息列表
    """
    if not messages:
        return []

    # 先过滤掉所有紧跟在tool消息后面的user消息
    filtered_messages = []
    i = 0
    while i < len(messages):
        curr_msg = messages[i]
        if curr_msg.get("role") == "tool":
            filtered_messages.append(curr_msg)
            i += 1
            while i < len(messages) and messages[i].get("role") == "user":
                print(f"检测到tool消息后跟随user消息，舍弃第{i}条user消息")
                i += 1
        else:
            # 正常添加当前消息
            filtered_messages.append(curr_msg)
            i += 1

    return filtered_messages


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, web browsing, or human interaction (only for extreme cases), you can handle it all.The initial directory is: /Users/zhaozhen05/code/EvalManus/workspace",
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": '\n        CURRENT PLAN STATUS:\n        Plan: 五一新疆旅游规划表 (ID: plan_1746703324)\n======================================\n\nProgress: 0/7 steps completed (0.0%)\nStatus: 0 completed, 1 in progress, 0 blocked, 6 not started\n\nSteps:\n0. [→] 确定旅行日期和天数\n1. [ ] 预订机票和住宿\n2. [ ] 规划行程路线（如乌鲁木齐、喀纳斯、吐鲁番等）\n3. [ ] 准备旅行装备和必需品\n4. [ ] 了解当地天气和文化习俗\n5. [ ] 购买旅游保险\n6. [ ] 确认行程细节并打印相关文件\n\n\n        YOUR CURRENT TASK:\n        You are now working on step 0: "确定旅行日期和天数"\n\n        Please execute this step using the appropriate tools. When you\'re done, provide a summary of what you accomplished.\n        ',
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": "\nBased on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.\n\nBe sure to keep in mind to call the `terminate` tool/function call to end the task!!! If you want to stop the interaction at any point, use the `terminate` tool/function call.\n",
            "model": "deepseek-v3",
        },
        {
            "role": "assistant",
            "content": 'To complete Step 0 ("确定旅行日期和天数") of the travel plan, we need to determine the travel dates and the number of days for the trip. Since this involves gathering information, I can assist in the following ways:1. **Research the best travel dates for Xinjiang during the May Day holiday (五一假期)**.2. **Suggest a typical duration for a Xinjiang trip** (e.g., 7-10 days is common for covering major attractions like Urumqi, Kanas, and Turpan).Let me start by researching the May Day holiday dates and typical travel durations for Xinjiang.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>browser_use```json{"action":"web_search","query":"2024年五一假期日期"}```<｜tool▁call▁end｜>',
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": "\nBased on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.\n\nBe sure to keep in mind to call the `terminate` tool/function call to end the task!!! If you want to stop the interaction at any point, use the `terminate` tool/function call.\n",
            "model": "deepseek-v3",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "6fda6bf95efa4a658cdf2b7a337e7805",
                    "type": "function",
                    "function": {
                        "name": "browser_use",
                        "arguments": '{"action":"web_search","query":"2024年五一假期日期"}',
                    },
                }
            ],
            "model": "deepseek-v3",
        },
        {
            "role": "tool",
            "content": "Observed output of cmd `browser_use` executed:\nError: Browser action 'web_search' failed: BrowserType.launch: Executable doesn't exist at /Users/zhaozhen05/Library/Caches/ms-playwright/chromium-1161/chrome-mac/Chromium.app/Contents/MacOS/Chromium\n╔════════════════════════════════════════════════════════════╗\n║ Looks like Playwright was just installed or updated.       ║\n║ Please run the following command to download new browsers: ║\n║                                                            ║\n║     playwright install                                     ║\n║                                                            ║\n║ <3 Playwright Team                                         ║\n╚════════════════════════════════════════════════════════════╝",
            "name": "browser_use",
            "tool_call_id": "6fda6bf95efa4a658cdf2b7a337e7805",
            "model": "deepseek-v3",
        },
        {
            "role": "assistant",
            "content": "It seems there's an issue with the browser tool due to missing dependencies. Let me proceed with an alternative approach by manually providing the information based on general knowledge.### Step 0: Determine Travel Dates and Duration1. **May Day Holiday Dates (2024)**:   - The May Day holiday in China typically spans **May 1st to May 5th**, with potential adjustments for weekends. For 2024, the exact dates can be confirmed as **May 1 (Wednesday) to May 5 (Sunday)**.2. **Suggested Travel Duration for Xinjiang**:   - A typical trip to Xinjiang, covering major attractions like Urumqi, Kanas, and Turpan, usually takes **7-10 days**. This allows enough time for travel between destinations and exploration.### Next Steps:- **Confirm the exact dates** (May 1-5, 2024) and decide on the trip duration (e.g., 7 days).- Proceed to **Step 1: Book flights and accommodations** based on the confirmed dates.Would you like me to proceed with Step 1, or do you need any adjustments to the dates/duration? Otherwise, I can terminate the task.",
            "tool_calls": [
                {
                    "id": "85c7b412fd444d59900891cf66629838",
                    "type": "function",
                    "function": {
                        "name": "terminate",
                        "arguments": '{"status":"success"}',
                    },
                }
            ],
            "model": "deepseek-v3",
        },
        {
            "role": "tool",
            "content": "Observed output of cmd `terminate` executed:\nThe interaction has been completed with status: success",
            "name": "terminate",
            "tool_call_id": "85c7b412fd444d59900891cf66629838",
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": "\nBased on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.\n\nBe sure to keep in mind to call the `terminate` tool/function call to end the task!!! If you want to stop the interaction at any point, use the `terminate` tool/function call.\n",
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": '\n        CURRENT PLAN STATUS:\n        Plan: 五一新疆旅游规划表 (ID: plan_1746703324)\n======================================\n\nProgress: 1/7 steps completed (14.3%)\nStatus: 1 completed, 1 in progress, 0 blocked, 5 not started\n\nSteps:\n0. [✓] 确定旅行日期和天数\n1. [→] 预订机票和住宿\n2. [ ] 规划行程路线（如乌鲁木齐、喀纳斯、吐鲁番等）\n3. [ ] 准备旅行装备和必需品\n4. [ ] 了解当地天气和文化习俗\n5. [ ] 购买旅游保险\n6. [ ] 确认行程细节并打印相关文件\n\n\n        YOUR CURRENT TASK:\n        You are now working on step 1: "预订机票和住宿"\n\n        Please execute this step using the appropriate tools. When you\'re done, provide a summary of what you accomplished.\n        ',
            "model": "deepseek-v3",
        },
        {
            "role": "user",
            "content": "\nBased on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.\n\nBe sure to keep in mind to call the `terminate` tool/function call to end the task!!! If you want to stop the interaction at any point, use the `terminate` tool/function call.\n",
            "model": "deepseek-v3",
        },
    ]

    formatted_messages = format_ernie_messages(messages)
    for msg in formatted_messages:
        print(msg.get("role"))
