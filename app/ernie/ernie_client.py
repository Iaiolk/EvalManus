import json
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

import requests

# 全局变量，用于跟踪当前工具使用ID
CURRENT_TOOLUSE_ID = None


# OpenAI风格响应格式处理类
class OpenAIResponse:
    def __init__(self, data):
        # 递归将嵌套的字典和列表转换为OpenAIResponse对象
        for key, value in data.items():
            if isinstance(value, dict):
                value = OpenAIResponse(value)
            elif isinstance(value, list):
                value = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]
            setattr(self, key, value)

    def model_dump(self, *args, **kwargs):
        # 将对象转换为字典并添加时间戳
        data = self.__dict__
        data["created_at"] = datetime.now().isoformat()
        return data


# 主客户端类，用于与百度千帆平台交互
class ErnieClient:
    def __init__(self, api_key=None, model_api=None):
        # 设置默认API和密钥
        self.api_key = (
            api_key
            or "Bearer bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"
        )
        self.model_api = (
            model_api or "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
        )
        self.model_api_acc = "https://qianfan.baidubce.com/v2/chat/completions"
        self.chat = Chat(self)


# 聊天接口类
class Chat:
    def __init__(self, client):
        self.completions = ChatCompletions(client)


# 核心类，处理聊天完成功能
class ChatCompletions:
    def __init__(self, client):
        self.client = client

    def _convert_openai_tools_to_ernie_format(self, tools):
        # 将OpenAI函数调用格式转换为文心一言工具格式
        ernie_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                ernie_tool = {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
                ernie_tools.append(ernie_tool)
        return ernie_tools

    def _convert_openai_messages_to_ernie_format(self, messages):
        # 将OpenAI消息格式转换为文心一言消息格式
        ernie_messages = []
        for message in messages:
            if message.get("role") == "system":
                # 文心一言将系统消息作为用户消息的前缀
                ernie_messages.append(
                    {"role": "user", "content": f"系统指令：{message.get('content')}"}
                )
            elif message.get("role") in ["user", "assistant"]:
                ernie_message = {
                    "role": message.get("role"),
                    "content": message.get("content"),
                }
                ernie_messages.append(ernie_message)
            elif message.get("role") == "tool":
                # 工具响应处理
                global CURRENT_TOOLUSE_ID
                ernie_message = {
                    "role": "assistant",
                    "content": f"工具调用结果: {message.get('content')}",
                }
                ernie_messages.append(ernie_message)
        return ernie_messages

    def _convert_ernie_response_to_openai_format(self, ernie_response, knowledge=None):
        # 将文心一言响应格式转换为OpenAI格式
        try:
            print(f"DEBUG - 收到的原始响应: {ernie_response}")
            response_data = json.loads(ernie_response)

            # 百度文心一言API可能有多种响应格式，尝试处理常见格式
            result = ""
            if "result" in response_data:
                result = response_data.get("result", "")
            elif "data" in response_data:
                result = response_data["data"].get("result", "")
            elif "content" in response_data:
                result = response_data.get("content", "")
            elif "message" in response_data and "content" in response_data["message"]:
                result = response_data["message"].get("content", "")

            print(f"DEBUG - 解析出的结果: {result}")

            # 构建最终OpenAI格式响应
            openai_format = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "created": int(time.time()),
                "object": "chat.completion",
                "system_fingerprint": None,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": result,
                            "role": "assistant",
                            "tool_calls": None,
                            "function_call": None,
                        },
                    }
                ],
                "usage": {
                    "completion_tokens": response_data.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                    "prompt_tokens": response_data.get("usage", {}).get(
                        "prompt_tokens", 0
                    ),
                    "total_tokens": response_data.get("usage", {}).get(
                        "total_tokens", 0
                    ),
                },
            }
            return OpenAIResponse(openai_format)
        except Exception as e:
            print(f"Error converting Ernie response: {e}")
            print(f"Problematic response: {ernie_response}")
            # 返回一个错误响应
            return OpenAIResponse(
                {
                    "id": f"error-{uuid.uuid4()}",
                    "created": int(time.time()),
                    "object": "chat.completion",
                    "choices": [
                        {
                            "finish_reason": "error",
                            "index": 0,
                            "message": {
                                "content": f"Error processing response: {e}",
                                "role": "assistant",
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

    def _invoke_ernie(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        knowledge: Optional[List[dict]] = None,
        **kwargs,
    ) -> OpenAIResponse:
        # 非流式调用文心一言模型
        ernie_messages = self._convert_openai_messages_to_ernie_format(messages)
        url = self.client.model_api

        # 尝试使用标准API端点
        payload = {
            "messages": ernie_messages,
            "stream": False,
            "search_mode": "required",
            "response_format": "text",
            "enable_deep_search": False,
            "model": model or "deepseek-r1",
            "enable_reasoning": True,
            "resource_type_filter": [{"type": "web", "top_k": 10}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if knowledge:
            payload["additional_knowledge"] = knowledge

        if tools:
            payload["tools"] = self._convert_openai_tools_to_ernie_format(tools)

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.client.api_key,
        }

        try:
            print(f"DEBUG - 发送请求到: {url}")
            print(f"DEBUG - 请求头: {headers}")
            print(f"DEBUG - 请求体: {json.dumps(payload)}")

            response = requests.post(url, headers=headers, data=json.dumps(payload))

            print(f"DEBUG - 响应状态码: {response.status_code}")

            # 如果第一个API调用失败，尝试使用备用API端点
            if response.status_code != 200 or not response.text or response.text == "":
                print("DEBUG - 主API调用失败，尝试备用API端点")
                url = self.client.model_api_acc

                # 调整备用API的参数
                fallback_payload = {
                    "messages": ernie_messages,
                    "stream": False,
                    "model": model or "ERNIE-Bot-4",
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }

                response = requests.post(
                    url, headers=headers, data=json.dumps(fallback_payload)
                )
                print(f"DEBUG - 备用API响应状态码: {response.status_code}")

            if not response.text or response.text.strip() == "":
                return OpenAIResponse(
                    {
                        "id": f"error-{uuid.uuid4()}",
                        "created": int(time.time()),
                        "object": "chat.completion",
                        "choices": [
                            {
                                "finish_reason": "error",
                                "index": 0,
                                "message": {
                                    "content": "API返回了空响应",
                                    "role": "assistant",
                                },
                            }
                        ],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                )

            return self._convert_ernie_response_to_openai_format(
                response.text, knowledge
            )
        except Exception as e:
            print(f"Error calling Ernie API: {e}")
            return OpenAIResponse(
                {
                    "id": f"error-{uuid.uuid4()}",
                    "created": int(time.time()),
                    "object": "chat.completion",
                    "choices": [
                        {
                            "finish_reason": "error",
                            "index": 0,
                            "message": {
                                "content": f"API call failed: {e}",
                                "role": "assistant",
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

    def _invoke_ernie_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        knowledge: Optional[List[dict]] = None,
        **kwargs,
    ) -> OpenAIResponse:
        # 流式调用文心一言模型
        ernie_messages = self._convert_openai_messages_to_ernie_format(messages)
        url = self.client.model_api

        payload = {
            "messages": ernie_messages,
            "stream": True,
            "search_mode": "required",
            "response_format": "text",
            "enable_deep_search": False,
            "model": model or "deepseek-r1",
            "enable_reasoning": True,
            "resource_type_filter": [{"type": "web", "top_k": 10}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if knowledge:
            payload["additional_knowledge"] = knowledge

        if tools:
            payload["tools"] = self._convert_openai_tools_to_ernie_format(tools)

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.client.api_key,
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), stream=True
            )

            if response.status_code != 200:
                print(f"DEBUG - 流式API调用失败，状态码: {response.status_code}")
                return OpenAIResponse(
                    {
                        "id": f"error-{uuid.uuid4()}",
                        "created": int(time.time()),
                        "object": "chat.completion",
                        "choices": [
                            {
                                "finish_reason": "error",
                                "index": 0,
                                "message": {
                                    "content": f"Stream API call failed with status: {response.status_code}",
                                    "role": "assistant",
                                },
                            }
                        ],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                )

            # 初始化响应结构
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    print(f"DEBUG - 收到流式行: {decoded_line}")
                    if decoded_line.startswith("data:"):
                        try:
                            data = json.loads(decoded_line[5:])
                            if "result" in data:
                                chunk = data["result"]
                                full_response += chunk
                                print(chunk, end="", flush=True)
                        except json.JSONDecodeError as e:
                            print(f"DEBUG - JSON解析错误: {e}, 原始行: {decoded_line}")
            print()

            # 构造完整响应
            if not full_response:
                print("DEBUG - 流式响应中没有内容")

            complete_response = json.dumps({"result": full_response})
            return self._convert_ernie_response_to_openai_format(complete_response)
        except Exception as e:
            print(f"Error in streaming response: {e}")
            return OpenAIResponse(
                {
                    "id": f"error-{uuid.uuid4()}",
                    "created": int(time.time()),
                    "object": "chat.completion",
                    "choices": [
                        {
                            "finish_reason": "error",
                            "index": 0,
                            "message": {
                                "content": f"Stream API call failed: {str(e)}",
                                "role": "assistant",
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: Optional[bool] = False,
        tools: Optional[List[dict]] = None,
        knowledge: Optional[List[dict]] = None,
        **kwargs,
    ) -> OpenAIResponse:
        # 聊天完成的主入口点
        if stream:
            return self._invoke_ernie_stream(
                model,
                messages,
                max_tokens,
                temperature,
                tools,
                knowledge,
                **kwargs,
            )
        else:
            return self._invoke_ernie(
                model,
                messages,
                max_tokens,
                temperature,
                tools,
                knowledge,
                **kwargs,
            )


# 兼容原始API的函数
def model_res(prompt, knowledge=None):
    """
    请求模型 (兼容原始API)
    """
    # 直接使用原始实现方式，确保兼容性
    url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
    api_key = "Bearer bce-v3/ALTAK-pOhfMH708hRvFf1pSDthj/141ada82dd49207be48d9bd013234550f9e04805"

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
            "additional_knowledge": knowledge or [],
        }
    )
    headers = {"Content-Type": "application/json", "Authorization": api_key}

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        print(f"DEBUG - model_res响应状态码: {response.status_code}")
        print(f"DEBUG - model_res响应内容: {response.text}")
        return response.text
    except Exception as e:
        print(f"Error in model_res: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # 演示两种使用方式

    # 方式1: 使用兼容原始API的函数
    prompt = "请给我推荐一些适合初学者的Python学习资源。"
    knowledge = []
    response = model_res(prompt, knowledge)
    print("原始API方式调用结果:")
    print(response)
    print("\n" + "-" * 50 + "\n")

    # 方式2: 使用OpenAI风格的API
    client = ErnieClient()
    messages = [
        {"role": "user", "content": "请给我推荐一些适合初学者的Python学习资源。"}
    ]
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        stream=False,
    )
    print("OpenAI风格API调用结果:")
    print(
        response.choices[0].message.content
        if hasattr(response, "choices") and response.choices
        else "无结果"
    )
