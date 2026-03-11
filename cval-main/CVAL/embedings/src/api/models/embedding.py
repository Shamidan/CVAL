from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from api.models._base import fields
from api.models.detection import FramePrediction


@fields(
    'embedding_id: Optional[str]',
    'embedding: List[float]',
)
class EmbeddingModel(BaseModel):
    embedding_id: Optional[str]
    embedding: List[float]


@fields(
    'embeddings: List[EmbeddingModel]',
    'frame_id: str'
)
class FrameEmbeddingModel(BaseModel):
    embeddings: List[EmbeddingModel]
    frame_id: str


@fields(
    'frame_id: str',
    'embeddings_quantity: int',
    'embeddings: List[str]'
)
class FrameEmbeddingResponseModel(BaseModel):
    frame_id: str
    embeddings_quantity: int
    embeddings: List[str]


@fields(
    'frames_quantity: int',
    'frames: Union[List[FrameEmbeddingResponseModel], List]'
)
class EmbeddingsMetaResponse(BaseModel):
    frames_quantity: int
    frames: Union[List[FrameEmbeddingResponseModel], List]