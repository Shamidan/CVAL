import uvicorn

from src.get_app import get_app

APP = get_app()

if __name__ == '__main__':
    uvicorn.run(APP,
                host='0.0.0.0',
                port=8000)
