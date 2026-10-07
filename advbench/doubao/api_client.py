# -*- coding: utf-8 -*-
"""
API 客户端 - 延迟初始化
判断越狱也使用 doubao
"""
import time
from openai import OpenAI
import config


class APIClient:
    """API客户端，支持延迟初始化和重试"""

    def __init__(self, config_dict):
        self.config = config_dict
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                timeout=self.config["timeout"]
            )
        return self._client

    def call_with_retry(self, messages, temperature=0.7, max_tokens=8192):
        """带重试的API调用"""
        max_retries = self.config.get("max_retries", 3)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                if not response.choices:
                    return None, "API返回空choices列表"

                choice = response.choices[0]
                finish_reason = getattr(choice, 'finish_reason', None)

                if finish_reason == "length":
                    return None, "输出因长度限制被截断"
                if finish_reason and finish_reason not in ["stop", "length"]:
                    return None, f"非正常结束: finish_reason={finish_reason}"

                content = getattr(choice.message, 'content', None)
                if content is None:
                    return None, "API返回的content为None"

                if not isinstance(content, str):
                    return None, f"API返回的content不是字符串: {type(content).__name__}"

                content = content.strip()
                if not content:
                    return None, "API返回空字符串"

                return content, None

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep((attempt + 1) * 0.5)
                    continue

        return None, f"{type(last_error).__name__}: {last_error!s}"


# 延迟初始化客户端
_doubao_client = None
# 保留但不再使用
_chatgpt_client = None


def get_doubao_client():
    global _doubao_client
    if _doubao_client is None:
        _doubao_client = APIClient(config.get_doubao_config())
    return _doubao_client


def get_chatgpt_client():
    """保留兼容性，但实际不再使用"""
    global _chatgpt_client
    if _chatgpt_client is None:
        _chatgpt_client = APIClient(config.get_chatgpt_config())
    return _chatgpt_client


def call_doubao(messages=None, prompt=None):
    """调用doubao API - 用于回答问题"""
    if prompt:
        messages = [{"role": "user", "content": prompt}]
    if not messages:
        return None, "未提供消息"

    client = get_doubao_client()
    return client.call_with_retry(messages, temperature=0.7, max_tokens=8192)


def call_judge(prompt):
    """调用doubao判断越狱（不再使用ChatGPT）"""
    if not prompt:
        return None, "未提供提示词"

    messages = [{"role": "user", "content": prompt}]
    client = get_doubao_client()
    # 判断使用更低的温度，让结果更稳定
    return client.call_with_retry(messages, temperature=0.1, max_tokens=512)