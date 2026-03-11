import contextlib
from typing import AsyncIterator
from fastapi import FastAPI
import uvicorn

from src.integrations.di_stubs.stubs import KVStub, SQLSessionGetterStub
from src.storage.kv.types import KVStorageAIORedis, KVStorageFiles
from src.storage.sql.core.db_manager import db_manager, get_session
from src.storage.sql.core.settings import Settings
from src.api.router import MainView


def get_app(settings):
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db_manager.init(settings.database_url)
        yield
        await db_manager.close()

    api = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url='/')
    api.include_router(MainView().router, )
    api.dependency_overrides[KVStub] = KVStub(KVStorageFiles('./data'))
    api.dependency_overrides[SQLSessionGetterStub] = SQLSessionGetterStub(get_session)
    return api


SETTINGS = Settings()
API = get_app(SETTINGS)


if __name__ == "__main__":
    uvicorn.run(
        API,
        host=SETTINGS.app_host,
        port=SETTINGS.app_port,
    )

