import hashlib
import io
from pprint import pprint
from typing import List

import uvicorn

from bus.http.sync_client import BusHTTP
from fastapi import FastAPI, Request, APIRouter

from filter.filter_client import FilterService
from annotation.cvat_hook.hook_utils.handler_utils import create_zip
from annotation.cvat_hook.hook_utils.hook_clasess import Handler, EventWrapper, CVATProces
from cvat_settings import CVATSettings, ClientSettings
from filter.filter_funk import cls_funk_2
from abc_types import AnnotationServiceHookProto

app = FastAPI(docs_url='/')
router = APIRouter()


def subtract_lists(all_files, data_files):
    result = []
    for item in all_files:
        if item not in data_files:
            result.append(item)
    return result


def generate_file_hash(file_data: io.BytesIO) -> str:
    file = file_data.getvalue()
    return hashlib.sha256(file).hexdigest()


class AnnotationServiceHook(AnnotationServiceHookProto):

    async def send_data(self, route, images: list, annotation=None, task_config=None):
        zip_images = create_zip(images, zip_name='images.zip')

        with open(zip_images, 'rb') as zip_file:
            file = zip_file.read()
        zip_buffer = io.BytesIO(file)
        file_hashes_all = []
        for image in images:
            img_hash = generate_file_hash(image)
            file_hashes_all.append(img_hash)

        if route == 'al':
            upload_response = self.bus.upload_files(zip_buffer)

            init_response = self.bus.init_annotation(route, file_hashes_all)
            return {
                'upload_response': upload_response,
                'init_response': init_response
            }

        elif route == 'files':
            response_1 = self.bus.upload_files(zip_buffer)
            file_hashes_annotated = []

            for key in annotation.keys():
                file_hashes_annotated.append(key)

            annotation_zip = create_zip(annotation_data=annotation, zip_name='annotation.zip')
            with open(annotation_zip, 'rb') as annotation:
                anat = annotation.read()

            init_response = self.bus.init_annotation(route, file_hashes_all)
            # init_response = self.bus.init_annotation(route, file_hashes_annotated)
            byts_anat = io.BytesIO(anat)
            response_2 = self.bus.upload_annotation(route, byts_anat)

        elif route == 'filter':

            self.filter.send_data_to_annotation_client(zip_images, task_config=task_config)
            # TODO: Проверить как с этой движухой работает
            self.bus.upload_files(zip_buffer)
            self.bus.init_annotation('filter', file_hashes_all)


    @router.post('/webhook/')
    async def listen(self, request: Request):
        payload = await request.json()
        if payload['event'] == 'ping':
            print('ping')
            return 'ping'

        event_type = self.event_wrapper.detect_event_type(payload)
        if event_type in ['al', 'files']:

            task_data = payload.get('task', {})
            task_id = task_data.get('id')
            func = self.handlers[event_type]
            images, annotation = func(task_id)

            response = await self.send_data(event_type, images, annotation=annotation)
            return response

        elif event_type == 'filter':

            task_data = payload.get('task', {})
            task_name = task_data.get('name')
            task_name = task_name[:-8] + "[files]"
            task_id = task_data.get('id')
            project_id = task_data.get('project_id')
            func = self.handlers[event_type]
            images, _ = func(task_id)

            filtered_images = self.filter.filtering(images)

            task_config = {"name": f"{task_name}", 'project_id': project_id, 'image_quality': 100}

            await self.send_data(event_type, filtered_images, task_config=task_config)

        else:
            return 'Bad event type'


def tt_func(data: List[io.BytesIO]):
    return data

bus_url = f'{ClientSettings().bus_url}/api'
print(bus_url)

bus = BusHTTP(bus_url)
cvat_settings = CVATSettings()
cvat = CVATProces(settings=cvat_settings)
handler = Handler(cvat)
handlers = {
    'filter': handler.handle_filter,
    'al': handler.handle_al,
    'files': handler.handle_files
}
filter = FilterService(bus, 'files', cls_funk_2, cvat)
wrapper = EventWrapper()
hook = AnnotationServiceHook(
    bus,
    handlers=handlers,
    event_wrapper=wrapper, filter=filter)

app.post("/webhook/")(hook.listen)
uvicorn.run(app.post("/webhook/")(hook.listen), port=8002)
