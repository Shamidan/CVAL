from functools import wraps
from typing import (
    Dict, Union, Type,
)

from typing import Any

from pydantic import BaseModel


class CodeResponse(Exception):
    def __init__(self, code: int):
        self.code = code

    def __str__(self):
        return f'Code({self.code})'

    def __hash__(self):
        return hash(self.__str__())


class ResponseModel(BaseModel):
    detail: Any


def exception_decorator_factory(
        exceptions: Dict[Union[Exception, Type[Exception]], CodeResponse]
):
    def _decorator(func):
        @wraps(func)
        async def wrap_exception(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except tuple(exceptions.keys()) as e:
                exception_type = type(e)
                if exception_type in exceptions:
                    raise exceptions[exception_type]
                else:
                    raise e

        return wrap_exception

    return _decorator


def validate(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        if result is None:
            raise CodeResponse(404)
        return result
    return wrapper
