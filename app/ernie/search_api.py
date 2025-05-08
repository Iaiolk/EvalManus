import json
import logging
import math
import os
import re
import sys
import time
import urllib
from collections import OrderedDict, defaultdict
from configparser import ConfigParser
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

model_api = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
model_api_acc_flag = "https://qianfan.baidubce.com/v2/chat/completions"
api_key = (
    "Bearer bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"
)
model_type = "ernie-4.5-8k-preview"
# model_type = "ernie-4.0-8k"


def llm_classify_return(llm_return):
    """
    大模型返回结论结构化写出
    """
    try:
        result = json.loads(llm_return)
    except:
        print("大模型返回结果无法用json解析: " + str(llm_return))
        return {}
    # 提取结果
    if "choices" in result:
        ret = result["choices"][0]["message"]["content"]
        # 发现返回的json有些会写为markdown
        pattern = r"```json\n(.*?)\n```"
        match = re.search(pattern, ret, re.DOTALL)
        if match:
            extracted_content = match.group(1)
        else:
            extracted_content = ret
    else:  # 返回的json里没有result
        print("llm_classify_return: llm返回结果中没有choices")
        return {}
    # 解析结果
    try:
        ret = json.loads(extracted_content)
    except Exception as e:
        print("大模型分类解析异常:" + str(e))  # 解析错误
        print("extracted_content: " + str(extracted_content))
        return {}
    # 结果写入
    return ret


def model_res(prompt, knowledge):
    """
    请求模型
    """
    url = model_api

    payload = json.dumps(
        {
            "messages": [{"content": prompt, "role": "user"}],
            "stream": False,
            "search_mode": "required",
            "response_format": "text",
            "enable_deep_search": False,
            "model": "deepseek-r1",
            "enable_reasoning": True,
            "resource_type_filter": [{"type": "web", "top_k": 10}],
            "additional_knowledge": knowledge,
        }
    )

    headers = {"Content-Type": "application/json", "Authorization": api_key}

    response = requests.request("POST", url, headers=headers, data=payload)
    # if response.status_code == 200:
    return response.text
    # return "查询结果错误"


def model_res_acc_flag(prompt):
    """
    请求接口
    """
    url = model_api_acc_flag

    payload = json.dumps(
        {
            "model": model_type,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.01,
        }
    )

    headers = {"Content-Type": "application/json", "Authorization": api_key}

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.text
    return "查询结果错误"


# def get_llm_chain_score(prompt_file, *prompt_args):
#     """
#     准备LLM链

#     Args:
#         prompt_file (str): prompt模板文件
#         *prompt_args (tuple): prompt模板变量

#     Returns:
#         LLMChain: LLM链对象
#     """
#     # 读取prompt
#     prompt_template = open(prompt_file, "r").read()
#     # prompt_template = prompt_file

#     prompt = PromptTemplate(input_variables=prompt_args, template=prompt_template)
#     yiyan_llm = YiYan_score()
#     llm_chain = LLMChain(llm=yiyan_llm,  prompt=prompt, verbose=False)
#     return llm_chain


def parse_url_res(llm_ref):
    try:
        json_data = llm_ref
    except:
        print("解析url结果失败")
        return None

    all_res = []
    for each_res in json_data:
        each_res_dict = {}
        if "title" in each_res:
            each_res_dict["标题"] = each_res["title"]
        if "content" in each_res:
            each_res_dict["摘要"] = each_res["content"]
        if "url" in each_res:
            each_res_dict["URL"] = each_res["url"]
        if "date" in each_res:
            each_res_dict["发布日期"] = each_res["date"]

        all_res.append(each_res_dict)

    all_res_str = json.dumps(all_res, indent=4, ensure_ascii=False)
    return all_res_str


def get_acc_flag(query):
    acc_flag = "-"
    acc_flag_res = "-"

    prompt_file = "/home/work/songxianyang/LLM-Session/Session_text_generate/prompt_file/是否需要准确性校验prompt.txt"
    prompt_template = open(prompt_file, "r").read()
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    prompt_text = prompt.format(query=query)
    llm_res = llm_classify_return(model_res_acc_flag(prompt_text))
    if len(llm_res) == 0:
        return acc_flag, acc_flag_res
    else:
        try:
            acc_flag = llm_res["是否有准确答案"]
            acc_flag_res = llm_res["原因"]
        except:
            print("判断是否需要准确性校验失败: " + str(llm_res))
            acc_flag = "-"
            acc_flag_res = "-"

    return acc_flag, acc_flag_res


def get_acc_res(query, session_text, llm_res, llm_ref):
    acc_flag = "-"
    acc_flag_res = "-"

    prompt_file = "/home/work/songxianyang/LLM-Session/Session_text_generate/prompt_file/准确溯源结果输出.txt"
    prompt_template = open(prompt_file, "r").read()
    prompt = PromptTemplate(
        input_variables=["query", "session_text", "llm_res", "llm_ref"],
        template=prompt_template,
    )
    prompt_text = prompt.format(
        query=query, session_text=session_text, llm_res=llm_res, llm_ref=llm_ref
    )
    llm_res = llm_classify_return(model_res_acc_flag(prompt_text))
    # if len(llm_res) == 0:
    #     return acc_flag, acc_flag_res


if __name__ == "__main__":
    # df_3w2 = pd.read_excel('/home/work/songxianyang/gpt4o_test/2w2_2w3/3w2_llm_res（merge外包）.xlsx', sheet_name='专家、外包、llm打分pv')
    # df_3w2['qid'] = df_3w2['qid'].astype(str)
    # query_list = sorted(list(set(df_3w2['query'])))

    all_res = []
    # for i, row in df_3w2.iterrows():
    # try:
    #     each_query = row['query']
    #     session_text = row['llm_pv_session行为']
    # except:
    #     print(str(i) + ' ' + '获取query和session行为失败')
    #     continue
    each_query = "抗日时山东三支队长是谁杨国夫在山东先后什么职务"
    session_text = "主动检索: 抗日时山东三支队长是谁杨国夫在山东先后什么职务\n———————————————————————————————————————\n| 展现第1位搜索结果, 该结果为AI智能生成\n| AI智能生成资源是利用大模型针对query及相关搜索结果自动生成的内容\n| 用户曝光该资源56.596秒\n———————————————————————————————————————\n——————————————————————————————————————————\n| 展现第2位搜索结果, 该结果为大家都在搜, 此类卡片会推送一批相关query\n——————————————————————————————————————————\n交互点击第1条搜索结果（阿拉丁卡片）\n用户从落地页返回结果列表页\n———————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源30.776秒\n———————————————————————————————\n点击第2位大家都在搜卡片进行激发检索\n用户从落地页返回结果列表页\n——————————————————————————————\n| 再次展现第2位搜索结果, 用户曝光该资源4.294秒\n——————————————————————————————\n"

    # if each_query == '你好星期六免费观看完整版2025':

    acc_flag, acc_flag_res = get_acc_flag(each_query)

    if acc_flag == "是":
        ai_search_content = "-"
        ai_reference = "-"
        know_ledge = []
        prompt_template = each_query
        prompt = PromptTemplate(input_variables=[], template=prompt_template)
        prompt_text = prompt.format()

        try:
            llm_res = model_res(prompt_text, know_ledge)
            res = json.loads(llm_res)
        except:
            print("模型AI搜索失败" + prompt_text + str(res))

        if "choices" in res and "references" in res:
            ai_search_content = res["choices"][0]["message"]["content"]
            ai_reference = parse_url_res(res["references"])
            # print(type(ai_search_content))
            # print(type(ai_reference))

        if ai_reference != "-" and ai_search_content != "-":
            get_acc_res(each_query, session_text, ai_search_content, ai_reference)
    else:
        print("该pv无需做准确性校验")

    #     logging.info(str(llm_res))
    #     try:
    #         res = json.loads(llm_res)
    #     except:
    #         res = {}
    #         logging.info('大模型调用失败')

    #     if len(res) != 0:
    #         try:
    #             ai_search_content = res['choices'][0]['message']['content']
    #         except:
    #             ai_search_content = '-'
    #             logging.info('回答汇总获取失败')
    #         try:
    #             ai_reference = json.dumps(res['references'], indent=4, ensure_ascii=False)
    #         except:
    #             ai_reference = "{}"
    #             logging.info('参考资料获取失败')

    #         each_query_res.append(each_query)
    #         each_query_res.append(ai_search_content)
    #         each_query_res.append(ai_reference)

    #         all_res.append(each_query_res)

    # all_res_df = pd.DataFrame(
    #     all_res,
    #     columns=['query', 'AI回答总结', '参考资料']
    # )
    # all_res_df.to_excel('/home/work/songxianyang/LLM-Session/Session_text_generate/accurate_tracing_res/3w2.xlsx', index=None)
