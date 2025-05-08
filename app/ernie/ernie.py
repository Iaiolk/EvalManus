import json

import requests

model_api = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
model_api_acc_flag = "https://qianfan.baidubce.com/v2/chat/completions"
api_key = (
    "Bearer bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"
)
model_type = "ernie-4.5-8k-preview"


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
    return response.text


if __name__ == "__main__":
    prompt = "你是谁"
    knowledge = []
    response = model_res(prompt, knowledge)
    print(response)
