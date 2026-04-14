import ollama
from typing import List, Dict 
from src.prompts import ALL_GROUPS
import json
import time
from src.classes import GroupResponse, InferenceResult
    

class LLMInference:
    def __init__(self, model_name: str, think:bool=False):
        self.model_name = model_name
        self.think = think
        ollama.chat(
            model=model_name,
            messages=[{"role" : "user", "content" : "Say Hi!"}],
            keep_alive="-1m"
        )

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
            start_time = time.time()
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role" : "user", "content" : prompt, "images": images}],
                think=self.think,
                format="json"
            )
            
            finish_time = time.time() - start_time
            raw_response = response.message.content
            assert raw_response is not None, "LLM did not return any content"

            try:
                parsed_response = json.loads(raw_response)
                output_classes = self._extract_true_classes(parsed_response)
            
            except (json.JSONDecodeError, TypeError, ValueError):
                output_classes = []

            groups.append(
                GroupResponse(
                    group_name=group.name,
                    raw_response=raw_response,
                    thinking=response.message.thinking if hasattr(response.message, "thinking") else None,
                    total_time=finish_time,
                    output_classes=output_classes,
                )
            )

        return InferenceResult(groups=groups)