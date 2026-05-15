from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
import json
import logging
import pickle
import os
import hashlib

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

_log = logging.getLogger(__name__)

_RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
)


def _is_retryable_status(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, _RETRYABLE)


class LLMCache:
    def __init__(self, model_name: str, cache_dir: str = "temp/llm_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        safe_name = model_name.replace("/", "__")
        self.cache_path = os.path.join(cache_dir, f"{safe_name}.bin")

        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache = self._load_cache()

    def _load_cache(self):
        """Loads the binary file if it exists; otherwise returns an empty dict."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                return {}
        return {}

    def _save_cache(self):
        """Writes the current cache dictionary to the binary file."""
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def _generate_key(self, prompt: str):
        """Hashes the prompt to create a standard-length dictionary key."""
        return hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()

    def get(self, messages: str):
        """Retrieves a response from the cache if it exists."""
        key = self._generate_key(messages)
        return self.cache.get(key, None)

    def set(self, messages: str, response: dict):
        """Saves a response to the cache and persists it to the file."""
        key = self._generate_key(messages)
        self.cache[key] = response
        self._save_cache()


class LLM:
    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str,
        no_cache: bool = False,
    ):
        if base_url is not None:
            self.client = OpenAI(
                base_url=base_url, api_key=api_key if api_key is not None else "none"
            )
        else:
            import os
            from dotenv import load_dotenv

            load_dotenv()
            api_key = "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"
            if api_key is None:
                raise ValueError("No NVIDIA-NIM-API-KEY set!")
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
        self.model = model

        self.no_cache_flag = no_cache
        if not self.no_cache_flag:
            self.cache = LLMCache(self.model)

    @retry(
        retry=retry_if_exception(_is_retryable_status),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )
    def _call_api(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            # timeout=60,
            # extra_body={
            #     "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
            # },
        )
        return completion

    def get_chat_completions(self, messages, verbose=False):
        if verbose:
            print(f"model prompt: {json.dumps(messages)}")
        result = None
        if self.no_cache_flag is False:
            res = self.cache.get(json.dumps(messages))
            if res:
                result = res
            else:
                result = self._call_api(messages).model_dump()
                if result is not None:
                    self.cache.set(json.dumps(messages), result)

        # reasoning = getattr(
        #     completion.choices[0].message, "reasoning", None
        # ) or getattr(completion.choices[0].message, "reasoning_content", None)
        # if reasoning:
        #     print(reasoning)
        if verbose:
            print(f"Response: {result}")
        return result
