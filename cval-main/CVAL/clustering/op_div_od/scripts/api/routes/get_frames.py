from fastapi import APIRouter

from api.core.op_div_od_main import return_files
from api.models.model import ClusteringWorkerModel

frames = APIRouter(
    tags=['frames']
)


@frames.post('/sampling')
async def get_frames(
    json: ClusteringWorkerModel,
):
    return return_files(json)
