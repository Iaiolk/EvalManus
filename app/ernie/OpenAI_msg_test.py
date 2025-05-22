# 请安装 OpenAI SDK : pip install openai
# apiKey 获取地址： https://console.bce.baidu.com/iam/#/iam/apikey/list
# 支持的模型列表： https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu

from openai import OpenAI

client = OpenAI(
    base_url="https://qianfan.baidubce.com/v2",
    api_key="bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805",
)
response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {"content": "抗日时山东三支队长是谁杨国夫在山东先后什么职务", "role": "user"}
    ],
    temperature=0.8,
    top_p=0.8,
    extra_body={
        "penalty_score": 1,
        "web_search": {"enable": True, "enable_trace": True},
    },
)
print(response)
