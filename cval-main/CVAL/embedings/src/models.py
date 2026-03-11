from typing import List
from pydantic import BaseModel, Field


class BBoxScores(BaseModel):
    category_id: int
    score: float
    probabilities: List[float]


class FramePrediction(BaseModel):
    frame_id: str
    predictions: List[BBoxScores]


class SamplingArguments(BaseModel):
    frames: List[FramePrediction]
    num_of_samples: int
    bbox_selection_policy: str
    selection_strategy: str
    sort_strategy: str
    probs_weights: List[float]

class EmbeddingPredictionModel(BaseModel):
    embedding: List[float]
    category_id: int
    score: float


class FrameModel(BaseModel):
    frame_id: str
    predictions: List[EmbeddingPredictionModel]

class ResponseModel(BaseModel):
    num_of_samples: int
    frames: List[FrameModel] = Field(min_items=1)