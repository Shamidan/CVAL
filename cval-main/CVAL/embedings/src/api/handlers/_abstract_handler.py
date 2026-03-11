from functools import wraps

from requests import Request, Session, Response

from api.utils.exceptions import (
    Forbidden,
    Conflict,
    NotFound,
    NotAcceptable,
    SchemaException,
    UnknownException,
)


class AbstractHandler(Request):
    __exceptions_chain__ = (
        Forbidden,
        Conflict,
        NotFound,
        NotAcceptable,
        SchemaException
    )

    def __init__(self, session: Session, sub: str = '', url='',):
        self.session = session
        self.sub = sub

        super().__init__(
            method=None,
            url=url,
            headers=session.headers,
            files=None,
            data=None,
            params=None,
            auth=None,
            cookies=None,
            hooks=None,
            json=None,
        )

    @staticmethod
    def pos_val(func):
        @wraps(func)
        def _(*args, **kwargs):
            if args:
                raise ValueError()
            return func(**kwargs)
        return _

    def _get(self, url: str, params=None, stream=False, json=None):
        self.url = url
        self.method = 'get'
        self.params = params
        self.stream = stream

    def _delete(self, url: str, params=None, json=None):
        self._get(url, params=params)
        self.method = 'delete'

    def _post(self, url: str, json=None, params=None, stream=False, files=None):
        self._get(url, params=params, stream=stream)
        self.method = 'post'
        self.json = json
        self.files = files

    def _put(self, url: str, json=None, params=None, stream=False, files=None):
        self._post(url, json, params, stream=stream, files=files)
        self.method = 'put'

    def _validate_response(self, resp: Response):
        if resp.status_code >= 400:
            for exc in self.__exceptions_chain__:
                exc().handle(resp)
            raise UnknownException((resp.json()) if resp.status_code != 500 else 'Internal Server Error :(')

    def send(self):
        resp = self.session.send(self.prepare())
        self._validate_response(resp)
        return resp