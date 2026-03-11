from typing import List

from pydantic import BaseModel, Field


class EmbeddingPredictionModel(BaseModel):
    embedding: List[float]
    category_id: int
    score: float


class FrameModel(BaseModel):
    frame_id: str
    predictions: List[EmbeddingPredictionModel]


class ClusteringWorkerModel(BaseModel):
    num_of_samples: int
    frames: List[FrameModel] = Field(min_items=1)
