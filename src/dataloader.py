
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from pydantic import BaseModel
import json
import cv2
import re
from tqdm import tqdm #type:ignore
from src.classes import Metadata


@dataclass
class DataloaderItem:
    frames : List[bytes]
    metadata : Metadata

VIDEO_DIR_PATTERN = re.compile(r"^video(\d+)$", re.IGNORECASE)
VIDEO_FILE_PATTERN = re.compile(r"^(video\d+)_([^.]+)\.mp4$", re.IGNORECASE)

class Dataloader:
    def __init__(self, dataset_path: str, seconds_jump: float, batch_size: int):
        self.dataset_path = Path(dataset_path)
        self.seconds_jump = seconds_jump
        self.batch_size = batch_size
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        if self.seconds_jump <= 0:
            raise ValueError("seconds_jump must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        
        self._current_video_key: Optional[Tuple[str, str]] = None
        self._current_video_capture: Optional[cv2.VideoCapture] = None

        self.items_plan: List[Metadata] = self._prepare_items_plan()

    def clear(self) -> None:
        if self._current_video_capture is not None:
            self._current_video_capture.release()
        self._current_video_capture = None
        self._current_video_key = None

    @staticmethod
    def _parse_time_to_seconds(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            raise ValueError(f"Invalid time value: {value}")

        parts = value.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds

        raise ValueError(f"Invalid time format: {value}")

    @staticmethod
    def _normalize_group_name(group_name: str) -> str:
        return re.sub(r"^group_\d+_", "", group_name)
        
    def _get_batch_instances(self, frames_timestamps: List[float], metadata_dict: Dict)->Dict[str, List[str]]:
        classes_by_group: Dict[str, List[str]] = {}

        for group_name, situations in metadata_dict.items():
            if not isinstance(situations, dict):
                continue

            normalized_group_name = self._normalize_group_name(str(group_name))
            detected: set[str] = set()

            for situation_name, situation_data in situations.items():
                if not isinstance(situation_data, dict):
                    continue

                if not bool(situation_data.get("value", False)):
                    continue

                intervals = situation_data.get("intervals", []) or []
                if not intervals:
                    detected.add(str(situation_name))
                    continue

                parsed_intervals: List[Tuple[float, float]] = []
                for interval in intervals:
                    if not isinstance(interval, list) or len(interval) != 2:
                        continue
                    start = self._parse_time_to_seconds(interval[0])
                    end = self._parse_time_to_seconds(interval[1])
                    if end >= start:
                        parsed_intervals.append((start, end))

                if any(start <= ts <= end for ts in frames_timestamps for start, end in parsed_intervals):
                    detected.add(str(situation_name))

            classes_by_group[normalized_group_name] = sorted(detected)

        return classes_by_group
        
        
    
    def _get_video_inferences(self, video_id, video_dir:Path, metadata_dict: dict) ->List[Metadata]:
        video_type = VIDEO_FILE_PATTERN.match(video_dir.name).group(2) #type:ignore
        video_capture = cv2.VideoCapture(str(video_dir))
        fps = float(video_capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        total_seconds = float(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
        in_batch_jump = self.seconds_jump / self.batch_size
        
        metadatas = []
        
        start_sec = 0.0
        while start_sec < total_seconds:
            frames_timestamps = [
                float(start_sec + i * in_batch_jump)
                for i in range(self.batch_size)
                if start_sec + i * in_batch_jump < total_seconds
            ]
            classes_for_times = self._get_batch_instances(frames_timestamps, metadata_dict)
            metadatas.append(
                Metadata(
                    video_id=video_id,
                    video_type=video_type,
                    first_frame_timestamp=start_sec,
                    frames_timestamps=frames_timestamps,
                    frames_indexes=[int(fps * ts) for ts in frames_timestamps],
                    classes=classes_for_times
                )
            )

            start_sec += self.seconds_jump
        
        video_capture.release()
        return metadatas    
            
        
    def _prepare_items_plan(self)->List[Metadata]:
        
        inferences = []
        
        for video_dir in tqdm(self.dataset_path.iterdir(), desc="Processing videos path for metadata"):
            
            if not video_dir.is_dir() or not VIDEO_DIR_PATTERN.match(video_dir.name):
                continue
            
            if not (video_dir/f"{video_dir.name}_anotado.txt").exists():
                continue
            
            with (video_dir/f"{video_dir.name}_anotado.txt").open("r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            
            available_video_files = [f for f in video_dir.iterdir() if f.is_file() and VIDEO_FILE_PATTERN.match(f.name)]
            
            for video_file in available_video_files:
                inferences.extend(self._get_video_inferences(video_dir.name, video_file, metadata_dict))
        
        return inferences
    
    def __getitem__(self, key):
        if key >= len(self.items_plan):
            raise IndexError("Dataloader index out of range")
        
        metadata = self.items_plan[key]
        video_capture = cv2.VideoCapture(str(self.dataset_path / f"{metadata.video_id}" / f"{metadata.video_id}_{metadata.video_type}.mp4"))
        frames = []
        
        for frame_index in metadata.frames_indexes:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video_capture.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())
        
        video_capture.release()
        
        return DataloaderItem(
            frames=frames,
            metadata=metadata
        )
        
    def __len__(self):
        return len(self.items_plan)
        
    def __iter__(self) -> Iterator[DataloaderItem]:
        try:
            for metadata in self.items_plan:
                target_key = (metadata.video_id, metadata.video_type)
                if self._current_video_key != target_key:
                    self.clear()
                    self._current_video_capture = cv2.VideoCapture(
                        str(self.dataset_path / f"{metadata.video_id}" / f"{metadata.video_id}_{metadata.video_type}.mp4")
                    )
                    self._current_video_key = target_key

                if self._current_video_capture is None:
                    continue

                frames = []

                for frame_index in metadata.frames_indexes:
                    self._current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = self._current_video_capture.read()
                    if ret:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frames.append(buffer.tobytes())

                yield DataloaderItem(
                    frames=frames,
                    metadata=metadata
                )
        finally:
            self.clear()
        
        