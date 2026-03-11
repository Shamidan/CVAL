from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.code_response.decorators import CodeResponse
from src.api.code_response.responses import RESPONSES


async def internal_server_exception_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except CodeResponse as e:
        return JSONResponse(
            dict(detail=RESPONSES.get(e.code)),
            e.code,
        )
    except Exception as e:
        e = CodeResponse(500)
        return JSONResponse(
            dict(detail=RESPONSES.get(e)),
            e.code,
        )
