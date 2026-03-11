import asyncio

import configparser as cfg
import uvicorn
from fastapi import FastAPI
from api.routes.get_frames import frames


def get_app(*routes):
    c = cfg.ConfigParser()
    c.read(".ini")
    _app = FastAPI(
        **c['PROD'],
    )
    for route in routes:
        _app.include_router(route)
    return _app


app = get_app(frames)

if __name__ == '__main__':
    asyncio.run(uvicorn.Server(uvicorn.Config(app, port=5004)).serve())
