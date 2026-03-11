import re
from functools import wraps

from classy_fastapi import Routable
from classy_fastapi import delete as _delete
from classy_fastapi import get as _get
from classy_fastapi import patch as _patch
from classy_fastapi import post as _post
from classy_fastapi import put as _put


BASE_API_ROUTER = '/api/'


def clean_path(path):
    cleaned_path = re.sub(r'/{2,}', '/', path)
    cleaned_path = re.sub(r'/$', '', cleaned_path)
    return cleaned_path


def routable_api_wrapper(router_method):
    """
    method wrapper for API
    """

    @wraps(router_method)
    def wrapped(route, *args, **kwargs):
        route = clean_path(BASE_API_ROUTER + route)

        def _wrap_routable_call(func):
            return router_method(
                route,
                response_model_exclude_none=True,
                response_model_exclude_unset=True,
                tags=[func.__qualname__.split('.')[0].replace('View', '')],  *args, **kwargs
            )(func)
        return _wrap_routable_call

    return wrapped


class BaseRoutable(Routable):
    """
    abs wrapper for Routable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@routable_api_wrapper
def post(route, *args, **kwargs):
    """
    abs wrapper for post-method
    """
    return lambda func: _post(route, *args, **kwargs)(func)


@routable_api_wrapper
def get(route, *args, **kwargs):
    """
    abs wrapper for get-method
    """
    return _get(route, *args, **kwargs)


@routable_api_wrapper
def put(route, *args, **kwargs):
    """
    abs wrapper for put-method
    """

    return _put(route, *args, **kwargs)


@routable_api_wrapper
def patch(route, *args, **kwargs):
    """
    abs wrapper for patch-method
    """
    return _patch(route, *args, **kwargs)


@routable_api_wrapper
def delete(route, *args, **kwargs):
    """
    abs wrapper for delete-method
    """
    return _delete(route, *args, **kwargs)
