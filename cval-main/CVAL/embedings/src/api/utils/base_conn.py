import atexit
from contextlib import suppress

from requests import Session


class BaseConn:
    _active_connections = []

    def __init__(self, user_api_key: str, sync: bool = True):
        self.session = Session()
        self.session.headers = {'Authorization': f"user_api_key:{user_api_key}"}
        self._active_connections.append(self)
        self.sync = sync
        atexit.register(self.close_all)

    @classmethod
    def close_all(cls):
        for connection in cls._active_connections:
            with suppress(Exception):
                connection.close()
        cls._active_connections.clear()

    def __del__(self):
        with suppress(Exception):
            self.session.close()
        del self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def close(self):
        self.session.close()