from typing import Literal

from fastapi import FastAPI

from uncertanly_cval_web_service.src.api.router import ALView


def get_app(framework: Literal['torch', 'numpy', 'tensorflow'] = 'numpy'):
    match framework:
        case 'torch':
            from uncertanly_cval_web_service.src.core.al._torch import al
        case 'numpy':
            from uncertanly_cval_web_service.src.core.al._np import al
        case 'tf':
            from uncertanly_cval_web_service.src.core.al._torch import al
    API = FastAPI(docs_url='/', title='AL')
    API.include_router(ALView(al_func=al).router)
    return API


