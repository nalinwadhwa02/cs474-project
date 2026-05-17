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
            api_key = os.environ["NVIDIA-NIM-API-KEY"]
            # api_key = (
            #     "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"
            # )
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

    def call_with_retry(
        self,
        build_messages_fn,
        parse_fn,
        verify_fn=None,
        max_retries: int = 3,
        logger=None,
        log_prefix: str = "",
    ) -> dict:
        """Retry loop: build messages → call API → parse → verify (optional).

        Args:
            build_messages_fn: (prior_attempts: list[dict]) -> list[dict]
            parse_fn:          (raw: str) -> dict | None
            verify_fn:         (parsed: dict) -> {"done": bool, "succeeded": bool,
                                                   "errors": list[str], "result": dict | None}
                               If None, a successful parse is treated as success.

        Returns dict with keys:
            raw, parsed, verification, succeeded, attempt_log, parse_error,
            messages (last attempt's messages), aborted
        """

        def _log(level, msg, *args):
            if logger:
                full = f"{log_prefix}  {msg}" if log_prefix else msg
                getattr(logger, level)(full, *args)

        prior_attempts: list[dict] = []
        attempt_log: list[dict] = []
        last_raw = ""
        last_parsed = None
        last_verification = None
        last_messages: list[dict] = []

        for attempt_num in range(max_retries):
            label = f"attempt {attempt_num + 1}/{max_retries}"
            rec: dict = {"attempt": attempt_num + 1}

            last_messages = build_messages_fn(prior_attempts)
            rec["messages"] = last_messages

            _log("info", "%s  calling model ...", label)
            _log(
                "debug",
                "%s  messages:\n%s",
                label,
                json.dumps(last_messages, indent=2, ensure_ascii=False),
            )

            api = self.call(last_messages)
            breakpoint()
            if api["error"]:
                _log("warning", "%s  %s", label, api["error"])
                rec.update(
                    {
                        "raw_response": "",
                        "parsed": None,
                        "verification": None,
                        "errors": [api["error"]],
                    }
                )
                attempt_log.append(rec)
                if not api["retryable"]:
                    _log("error", "non-retryable API error, aborting")
                    return {
                        "raw": "",
                        "parsed": None,
                        "verification": None,
                        "succeeded": False,
                        "attempt_log": attempt_log,
                        "parse_error": api["error"],
                        "messages": last_messages,
                        "aborted": True,
                    }
                prior_attempts.append({"raw_response": "", "errors": [api["error"]]})
                continue

            last_raw = api["content"]
            rec["raw_response"] = last_raw
            _log("info", "%s  response received, parsing ...", label)
            _log("debug", "%s  raw response:\n%s", label, last_raw)

            last_parsed = parse_fn(last_raw)
            if last_parsed is None:
                errors = ["could not parse a JSON object from the response"]
                _log("warning", "%s  JSON parse failed", label)
                rec.update({"parsed": None, "verification": None, "errors": errors})
                attempt_log.append(rec)
                prior_attempts.append({"raw_response": last_raw, "errors": errors})
                last_verification = None
                continue

            rec["parsed"] = last_parsed
            _log(
                "debug",
                "%s  parsed:\n%s",
                label,
                json.dumps(last_parsed, indent=2, ensure_ascii=False),
            )

            if verify_fn is None:
                rec.update({"verification": None, "errors": []})
                attempt_log.append(rec)
                return {
                    "raw": last_raw,
                    "parsed": last_parsed,
                    "verification": None,
                    "succeeded": True,
                    "attempt_log": attempt_log,
                    "parse_error": None,
                    "messages": last_messages,
                    "aborted": False,
                }

            vresult = verify_fn(last_parsed)
            last_verification = vresult["result"]
            rec.update({"verification": last_verification, "errors": vresult["errors"]})

            breakpoint()
            if vresult["done"]:
                attempt_log.append(rec)
                return {
                    "raw": last_raw,
                    "parsed": last_parsed,
                    "verification": last_verification,
                    "succeeded": vresult["succeeded"],
                    "attempt_log": attempt_log,
                    "parse_error": None,
                    "messages": last_messages,
                    "aborted": False,
                }

            for e in vresult["errors"]:
                _log("warning", "%s  verification error: %s", label, e)
            attempt_log.append(rec)
            prior_attempts.append(
                {"raw_response": last_raw, "errors": vresult["errors"]}
            )

        _log("warning", "all %d attempts exhausted", max_retries)
        return {
            "raw": last_raw,
            "parsed": last_parsed,
            "verification": last_verification,
            "succeeded": False,
            "attempt_log": attempt_log,
            "parse_error": (
                "could not extract JSON object" if last_parsed is None else None
            ),
            "messages": last_messages,
            "aborted": False,
        }

    def call(self, messages: list[dict]) -> dict:
        """Call the model and return a result dict with content, error, and retryable flag.

        Returns:
            {"content": str, "error": None, "retryable": False}  on success
            {"content": None, "error": str, "retryable": bool}   on failure
        """
        try:
            response = self.get_chat_completions(messages, verbose=True)
            breakpoint()
            if response is None:
                return {
                    "content": None,
                    "error": "model API returned no response",
                    "retryable": True,
                }
            return {
                "content": response["choices"][0]["message"]["content"].strip(),
                "error": None,
                "retryable": False,
            }
        except APIStatusError as exc:
            return {
                "content": None,
                "error": f"API error {exc.status_code}: {exc.message}",
                "retryable": exc.status_code in (429, 500, 502, 503, 504),
            }
        except (APIConnectionError, APITimeoutError) as exc:
            return {
                "content": None,
                "error": f"API connection/timeout error: {exc}",
                "retryable": True,
            }
