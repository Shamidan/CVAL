import os
from abc import ABC, abstractmethod

from aioredis import from_url
from pathlib import Path
from src.storage.kv.redis.utils import save_to_redis


class FileStorage(ABC):
    @abstractmethod
    def save(self, name: str, data: bytes): ...

    @abstractmethod
    def exists(self, name: str): ...

    @abstractmethod
    def get_buffer(self, name: str): ...

    @abstractmethod
    def flush(self): ...


class KVStorageAIORedis(FileStorage):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url

    async def _get_redis(self):
        redis = await from_url(self.redis_url)
        return redis

    async def save(self, name: str, data: bytes):
        redis = self._get_redis()
        return save_to_redis(name, data, redis)

    async def exists(self, name: str):
        return self._get_redis().exists(name)

    async def get_buffer(self, name: str):
        return self._get_redis().get(name)

    async def flush(self):
        await self._get_redis().flushdb()

class KVStorageFiles(FileStorage):
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
    async def save(self, name: str, data: bytes):
        with open(self.path / name, 'wb') as f:
            f.write(data)

    async def exists(self, name: str):
        return (self.path / name).exists()

    async def get_buffer(self, name: str):
        with open(self.path / name, 'r') as f:
            return f.read()

    async def flush(self):
        for f_name in os.listdir(self.path):
            (self.path / f_name).unlink(missing_ok=True)

