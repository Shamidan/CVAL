from pprint import pprint

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
from pydantic import BaseModel
import torch

from api.models.embedding import FrameEmbeddingModel, EmbeddingModel
from bus_sdk import BusHTTP
from api.get_embedings import train_siam, get_embeddings
import shutil

from models import (SamplingArguments, BBoxScores, FramePrediction, EmbeddingPredictionModel, FrameModel,
                        ResponseModel)
from utlis.clear_dir import clear_directory

app = FastAPI(docs_url='/')

UPLOAD_DIR = Path("uploaded_files")
MODEL_PATH = "end_pool.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
bus = BusHTTP('http://cval_bus:8001/api')
UPLOAD_DIR.mkdir(exist_ok=True)
clear_directory(UPLOAD_DIR)


def create_embedding_request(
        embeddings,
        sampling_args: SamplingArguments
) -> ResponseModel:
    embedding_dict = {}
    for embed in embeddings:
        frame_id = embed.frame_id
        embedding = embed.embeddings[0].embedding

        embedding_dict[frame_id] = embedding

    frames = []
    for frame in sampling_args.frames:
        frame_id = frame.frame_id
        predictions = []

        if frame_id in embedding_dict:
            predictions.append(
                EmbeddingPredictionModel(
                    embedding=embedding_dict[frame_id],
                    category_id=frame.predictions[0].category_id,
                    score=frame.predictions[0].score
                )
            )

        if predictions:
            frames.append(FrameModel(frame_id=frame_id, predictions=predictions))

    return ResponseModel(
        num_of_samples=sampling_args.num_of_samples,
        frames=frames
    )


@app.post("/full_process")
async def full_process(
    request: SamplingArguments,
    epochs: int = 1,
    dim: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
    pattern: str = "*.jpg",
):

    for i in request.frames:
            file_hash = i.frame_id
            file = bus.get_file_by_hash(file_hash)
            file_class = i.predictions[0].category_id
            file_path = UPLOAD_DIR / f'{file_hash}__{file_class}__info.jpg'
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
    print('Начало обучения')
    train_siam(
            path_to_bboxes=str(UPLOAD_DIR),
            epochs=epochs,
            dim=dim,
            batch_size=batch_size,
            shuffle=shuffle,
            device=DEVICE,
        )
    print('Получаем эмбединги и предсказания')
    embeddings = get_embeddings(
            path_to_crops=str(UPLOAD_DIR),
            dim=dim,
            batch_size=batch_size,
            shuffle=shuffle,
            pattern=pattern,
            device=DEVICE,
        )
    # print(embeddings[0])

    response_params = create_embedding_request(embeddings, request)
    response_params = response_params.json()

    response = requests.post('http://cval_op_div_od:5004/sampling', data=response_params, headers={"Content-Type": "application/json"})
    clear_directory(UPLOAD_DIR)

    return response.json()
