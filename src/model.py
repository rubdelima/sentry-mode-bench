import ollama
from typing import List
from src.prompts import ALL_GROUPS
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from src.classes import GroupResponse, InferenceResult
    

class LLMInference:
    def __init__(self, model_name: str, think:bool=False, timeout_sec: float = 25.0):
        self.model_name = model_name
        self.think = think
        self.timeout_sec = timeout_sec
        ollama.chat(
            model=model_name,
            messages=[{"role" : "user", "content" : "Say Hi!"}],
            keep_alive="-1m"
        )

    def _chat_with_timeout_retry(self, prompt: str, images: List[bytes]) -> tuple[str, str | None, float]:
        attempts = 2
        last_elapsed = 0.0

        for _ in range(attempts):
            start_time = time.time()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                ollama.chat,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": images}],
                think=self.think,
                format="json",
            )

            try:
                response = future.result(timeout=self.timeout_sec)
                elapsed = time.time() - start_time
                raw_response = response.message.content
                return (raw_response if raw_response is not None else "{}", getattr(response.message, "thinking", None), elapsed)
            except FutureTimeoutError:
                last_elapsed = time.time() - start_time
                future.cancel()
                print(f"Chat request timed out after {self.timeout_sec} seconds. Retrying...")
            except Exception:
                last_elapsed = time.time() - start_time
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        return "{}", None, last_elapsed

    @staticmethod
    def _extract_true_classes(parsed_response: object) -> List[str]:
        if not isinstance(parsed_response, dict):
            return []

        output: List[str] = []
        for key, value in parsed_response.items():
            is_true = value is True
            if isinstance(value, str):
                is_true = value.strip().lower() == "true"
            if isinstance(value, (int, float)):
                is_true = bool(value)

            if is_true:
                output.append(str(key))

        return sorted(set(output))

    def predict(self, images: List[bytes]) -> InferenceResult:
        groups: List[GroupResponse] = []

        for group in ALL_GROUPS:
            prompt = group.get_prompt()
            raw_response, thinking, finish_time = self._chat_with_timeout_retry(prompt, images)

            try:
                parsed_response = json.loads(raw_response)
                output_classes = self._extract_true_classes(parsed_response)
            
            except (json.JSONDecodeError, TypeError, ValueError):
                output_classes = []

            groups.append(
                GroupResponse(
                    group_name=group.name,
                    raw_response=raw_response,
                    thinking=thinking,
                    total_time=finish_time,
                    output_classes=output_classes,
                )
            )

        return InferenceResult(groups=groups)