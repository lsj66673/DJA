# -*- coding: utf-8 -*-
"""
API 客户端 - 延迟初始化
Qwen 回答问题 + GPT-5.4-mini 判断越狱
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

    def call_with_retry(self, messages, temperature=0.7, max_tokens=8192, use_completion_tokens=False):
        """带重试的API调用"""
        max_retries = self.config.get("max_retries", 3)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if use_completion_tokens:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens
                    )
                else:
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
_qwen_client = None
_judge_client = None


def get_qwen_client():
    """Qwen 客户端（回答问题）"""
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = APIClient(config.get_qwen_config())
    return _qwen_client


def get_judge_client():
    """判断模型客户端（GPT-5.4-mini）"""
    global _judge_client
    if _judge_client is None:
        _judge_client = APIClient(config.get_judge_config())
    return _judge_client


def call_qwen(messages=None, prompt=None):
    """调用 Qwen API - 用于回答问题"""
    if prompt:
        messages = [{"role": "user", "content": prompt}]
    if not messages:
        return None, "未提供消息"

    client = get_qwen_client()
    return client.call_with_retry(messages, temperature=0.7, max_tokens=8192, use_completion_tokens=False)


def call_judge(prompt):
    """调用 GPT-5.4-mini 判断越狱"""
    if not prompt:
        return None, "未提供提示词"

    messages = [{"role": "user", "content": prompt}]
    client = get_judge_client()
    # 判断使用更低的温度，让结果更稳定
    return client.call_with_retry(messages, temperature=0.1, max_tokens=512, use_completion_tokens=True)