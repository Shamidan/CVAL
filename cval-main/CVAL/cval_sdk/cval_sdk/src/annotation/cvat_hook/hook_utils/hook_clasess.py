import hashlib
from pprint import pprint
from typing import Iterable, Tuple, Any
import io
from re import search, findall
from cvat_sdk import Client
from cvat_sdk.core.proxies.tasks import ResourceType
from pydantic_settings import BaseSettings
import requests
from cvat_settings import CVATSettings
from abc_types import HandlerProto, EventWrapperProto, EventTypeLiteral, AnnotationServiceClientProto
from annotation.cvat_hook.hook_utils.handler_utils import create_zip
from PIL import Image


def create_image_points(image: io.BytesIO):
    image = Image.open(image)
    width = image.width
    height = image.height
    points = [0.0, 0.0, width, height]
    return points


def generate_file_hash(file_data: io.BytesIO) -> str:
    file = file_data.getvalue()
    return hashlib.sha256(file).hexdigest()


class Handler(HandlerProto):
    def handle_al(self, task_id):
        images, _ = self.annotation_service.download_data(task_id, 'al')
        return None, None, images

    def handle_filter(self, task_id):
        images, _ = self.annotation_service.download_data(task_id, 'filter')

        return images, None

    def handle_files(self, task_id):
        images, annotations = self.annotation_service.download_data(task_id, 'files')

        return images, annotations


class EventWrapper(EventWrapperProto):

    def detect_event_type(self, data, *args, **kwargs) -> tuple[Any, Any | None]:
        task_data = data.get('task', {})
        status = task_data.get('status')
        task_name_raw = task_data.get('name')
        task_route = self._extract_substring(task_name_raw)
        if status == 'completed':
            if task_route in ['filter', 'al', 'files']:
                return task_route

    def _extract_substring(self, text):
        match = search(r'\[(.*?)\]', text)
        if match:
            return match.group(1)
        return None


class CVATProces(AnnotationServiceClientProto):

    def __init__(self, settings: BaseSettings):
        super().__init__(settings)
        self.client = self._login_client()

    def download_data(self, task_id, mode):
        task = self.client.tasks.retrieve(task_id)
        task_name = str(task.name)[:-7]
        project_id = str(task.project_id)

        size = task.get_meta().size
        annotations = task.get_annotations()

        frames_list = {}

        for i in range(size):
            frame = task.get_frame(i)
            frames_list[i] = frame

        parsed_annotations = {}
        marked_frames = []
        frame_ids = self._get_all_frame_id(size)
        shapes = annotations.get('shapes')
        tags = annotations.get('tags')

        if shapes:

            for shape in shapes:
                frame_id = shape["frame"]
                frame = task.get_frame(frame_id, quality='original')
                frame_hash = generate_file_hash(frame)
                shape_type = shape.get('type')

                label_id = shape["label_id"]
                points = shape["points"]
                if frame not in parsed_annotations:
                    parsed_annotations[frame_hash] = {"boxes": [], "labels": [], 'task_name': '', 'project_id': ''}
                parsed_annotations[frame_hash]["boxes"].append(points)
                parsed_annotations[frame_hash]["labels"].append(label_id)
                parsed_annotations[frame_hash]["task_name"] = task_name
                parsed_annotations[frame_hash]["project_id"] = project_id
                marked_frames.append(frame)
        elif tags:
            for tag in tags:
                frame_id = tag["frame"]
                frame = task.get_frame(frame_id, quality='original')
                frame_hash = generate_file_hash(frame)

                label_id = tag["label_id"]
                if frame not in parsed_annotations:
                    parsed_annotations[frame_hash] = {"boxes": [], "labels": [], 'task_name': '', 'project_id': ''}
                parsed_annotations[frame_hash]["boxes"].append(create_image_points(frame))
                parsed_annotations[frame_hash]["labels"].append(label_id)
                parsed_annotations[frame_hash]["task_name"] = task_name
                parsed_annotations[frame_hash]["project_id"] = project_id
                marked_frames.append(frame)

        unmarked_frames = []
        for ind in frame_ids:
            frame = frames_list[ind]
            unmarked_frames.append(frame)

        if mode == 'filter':
            return frames_list.values(), []

        elif mode == 'files':
            return marked_frames, parsed_annotations

        elif mode == 'al':

            return unmarked_frames, []
        else:
            return 'Bad mode'

    def send_data_to_service(self, data, config, *args, **kwargs):
        new_task = self.client.tasks.create_from_data(
            spec=config,
            resource_type=ResourceType.LOCAL,
            resources=[data],
            data_params={'image_quality': 100}
        )

        return new_task.id

    def _login_client(self):
        client = Client(url=self.settings.cvat_url)
        client.login((self.settings.user_name, self.settings.password))
        return client

    def _get_all_frame_id(self, size):
        frame_ids = []
        for i in range(size):
            frame_ids.append(i)
        return frame_ids

    def get_tags(self, task_id, mode):
        task = self.client.tasks.retrieve(task_id)
        size = task.get_annotations()
        frame = task.get_frame(0)
        points = create_image_points(frame)

        return points

    def get_poligons(self, task_id, mode):
        task = self.client.tasks.retrieve(task_id)
        ann = task.get_annotations()

        return ann

