from requests import Response


class APIException(Exception):
    status_code: int
    msg: str = {'detail': 'error'}

    def handle(self, response: Response):
        if self.status_code == response.status_code:
            self.msg = response.json()
            raise self

    def __repr__(self):
        return f'{self.status_code}: {self.msg}'

    def __str__(self):
        return self.__repr__()


class UnknownException(Exception):
    pass


class Forbidden(APIException):
    status_code: int = 403


class NotFound(APIException):
    status_code: int = 404


class NotAcceptable(APIException):
    status_code: int = 406


class Conflict(APIException):
    status_code: int = 409


class SchemaException(APIException):
    status_code: int = 422