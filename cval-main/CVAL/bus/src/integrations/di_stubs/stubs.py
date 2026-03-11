from src.storage.kv.types import FileStorage


class KVStub:

    def __init__(self, kv_storage: FileStorage):
        self.kv_storage = kv_storage

    def __call__(self):
        return self.kv_storage


class SQLSessionGetterStub:
    def __init__(self, session_getter):
        self.session_getter = session_getter

    async def __call__(self):
        async for i in self.session_getter():
            yield i
