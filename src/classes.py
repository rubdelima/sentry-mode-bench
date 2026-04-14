from pydantic import BaseModel
from typing import List, Optional, Dict

class GroupResponse(BaseModel):
    group_name :str
    raw_response : str
    thinking : Optional[str] = None
    total_time: float
    output_classes : List[str]
    
class InferenceResult(BaseModel):
    groups: List[GroupResponse]

class Metadata(BaseModel):
    video_id: str
    video_type: str
    first_frame_timestamp: float
    frames_timestamps: List[float]
    frames_indexes : List[int]
    classes: Dict[str, List[str]]

class InferenceEntry(Metadata):
    inference: InferenceResult

class RunResult(BaseModel):
    model_name: str
    think: bool
    seconds_jump: float
    batch_size: int
    inferences: List[InferenceEntry]

class VideoResult(BaseModel):
    video_type : str
    inference_results : List[InferenceResult]

class VideoSetResult(BaseModel):
    video_name : str
    video_results : List[VideoResult]

class DatasetResult(BaseModel):
    model_name : str
    think : bool
    seconds_jump : float
    batch_size : int
    video_set_results : List[VideoSetResult]
