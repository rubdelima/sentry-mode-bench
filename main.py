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

def save_result(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def load_result(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)

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
    
    if output_path.exists():
        result = load_result(output_path)
    else:
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
        if len(result["inferences"]) > idx:
            idx += 1
            continue
        
        inference_result = model.predict(images=item.frames)
        result["inferences"].append(
            InferenceEntry(
                inference=inference_result,
                **item.metadata.model_dump()
            ).model_dump()
        )
        
        if idx % max(1, config["Search"]["save_checkpoints"]) == 0:
            save_result(result, output_path)
    
    dataloader.clear()
        
    save_result(result, output_path)


if __name__ == "__main__":
    main()
        
        