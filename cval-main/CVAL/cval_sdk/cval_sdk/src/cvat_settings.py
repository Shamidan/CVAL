from pydantic_settings import BaseSettings
from pathlib import Path

class CVATSettings(BaseSettings):

    user_name: str = "Eduard"
    password: str = "Baldej671MKSKOM"
    cvat_url: str = "http://cvat-server:8080"

    class Config:
        env_file = Path(__file__).parent / '.env'
        extra = 'ignore'

class ClientSettings(BaseSettings):


    bus_url: str = "http://cval_bus:8001"
    dataset_name: str = "Eshamidanov/test-dataset-viewer"
    project_id: int = 2

    class Config:
        env_file = Path(__file__).parent / '.env'
        extra = 'ignore'
