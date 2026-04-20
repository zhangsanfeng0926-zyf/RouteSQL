import json.decoder
import os
import socket

import openai
import httpx
from utils.enums import LLM
import time


OPENAI_CLIENT = None


def is_network_issue(exc: Exception) -> bool:
    err_name = exc.__class__.__name__.lower()
    msg = str(exc).lower()

    # Direct network/transport exceptions.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, socket.timeout, httpx.TimeoutException, httpx.NetworkError)):
        return True

    # Generic class-name based checks for SDK wrapped errors.
    name_keys = [
        "timeout",
        "connection",
        "network",
        "apierror",
        "servererror",
        "internalservererror",
        "ratelimit",
    ]
    if any(k in err_name for k in name_keys):
        return True

    # Message-based checks for transient network/service failures.
    msg_keys = [
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote end closed connection",
        "temporary failure",
        "name or service not known",
        "dns",
        "tls",
        "ssl",
        "502",
        "503",
        "504",
        "500",
        "too many connections",
        "rate limit",
    ]
    return any(k in msg for k in msg_keys)


def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    # if model == LLM.TONG_YI_QIAN_WEN:
    #     import dashscope
    #     dashscope.api_key = OPENAI_API_KEY
    # else:
    #     openai.api_key = OPENAI_API_KEY
    #     openai.organization = OPENAI_GROUP_ID
    global OPENAI_CLIENT
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    # openai>=1.0.0
    if hasattr(openai, "OpenAI"):
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_GROUP_ID:
            kwargs["organization"] = OPENAI_GROUP_ID
        if api_base:
            kwargs["base_url"] = api_base
        # Avoid inheriting OS/global proxy settings that may break TLS handshakes.
        kwargs["http_client"] = httpx.Client(trust_env=False)
        OPENAI_CLIENT = openai.OpenAI(**kwargs)
        return

    # openai<1.0.0
    openai.api_key = OPENAI_API_KEY
    if api_base:
        openai.api_base = api_base
    if OPENAI_GROUP_ID:
        openai.organization = OPENAI_GROUP_ID


def ask_completion(model, batch, temperature):
    if OPENAI_CLIENT is not None:
        response = OPENAI_CLIENT.completions.create(
            model=model,
            prompt=batch,
            temperature=temperature,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[";"]
        )
        response_clean = [choice.text for choice in response.choices]
        usage = response.usage
        return {
            "response": response_clean,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }

    response = openai.Completion.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def ask_chat(model, messages: list, temperature, n):
    system_message = {
        "role": "system",
        "content": "You are a Text-to-SQL engine. Return only one SQL query. Do not add explanation, markdown, or code fences."
    }
    merged_messages = [system_message] + messages

    if OPENAI_CLIENT is not None:
        response = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=merged_messages,
            temperature=temperature,
            max_tokens=200,
            n=n
        )
        response_clean = []
        for choice in response.choices:
            msg = choice.message
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text"):
                            text_parts.append(part["text"])
                    elif hasattr(part, "text") and getattr(part, "text"):
                        text_parts.append(getattr(part, "text"))
                content = "".join(text_parts)

            if (content is None or str(content).strip() == "") and hasattr(msg, "reasoning_content"):
                content = getattr(msg, "reasoning_content") or ""

            response_clean.append(content if isinstance(content, str) else str(content))
        usage = response.usage
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=merged_messages,
            temperature=temperature,
            max_tokens=200,
            n=n
        )
        response_clean = [choice["message"]["content"] for choice in response["choices"]]
        usage_dict = response["usage"]

    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        **usage_dict
    )


def ask_llm(model: str, batch: list, temperature: float, n:int):
    n_repeat = 0
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                assert n == 1
                response = ask_completion(model, batch, temperature)
            elif model in LLM.TASK_CHAT:
                # batch size must be 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]}]
                response = ask_chat(model, messages, temperature, n)
                response['response'] = [response['response']]
            break
        except Exception as e:
            if is_network_issue(e):
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for network issue: {e.__class__.__name__}", end="\n")
                time.sleep(20)
                continue
            if isinstance(e, json.decoder.JSONDecodeError):
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
                time.sleep(20)
                continue
            raise

    return response

