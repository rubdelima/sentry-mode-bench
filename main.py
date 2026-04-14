from src.constraints import config
from src.model import LLMInference
from src.dataloader import Dataloader
from pathlib import Path
import json
from src.classes import InferenceEntry
from tqdm import tqdm #type:ignore
from datetime import datetime
import re


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
    return sanitized or "model"

def main():
    model_name = config["Model"]["model_name"]
    batch_size = config["Video"]["batch_size"]
    model = LLMInference(
        model_name=model_name,
        think=config["Model"]["think"]
    )
    dataloader = Dataloader(
        dataset_path=config["Dataset"]["dataset_path"],
        seconds_jump=config["Video"]["seconds_jump"],
        batch_size=batch_size
    )

    safe_model_name = _safe_filename(model_name)
    output_path = Path("data") / f"{safe_model_name}_{batch_size}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model_name": model_name,
        "think": config["Model"]["think"],
        "seconds_jump": config["Video"]["seconds_jump"],
        "batch_size": batch_size,
        "timestamp": datetime.now().isoformat(),
        "inferences": []
    }
    idx = 0
    for item in tqdm(dataloader, desc="Processing inference entries"):
        inference_result = model.predict(images=item.frames)
        result["inferences"].append(
            InferenceEntry(
                inference=inference_result,
                **item.metadata.model_dump()
            ).model_dump()
        )
        idx += 1
        if idx == 2:
            break
    
    dataloader.clear()
        
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
        
        