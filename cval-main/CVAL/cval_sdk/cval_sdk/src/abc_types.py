import array
from abc import ABC, abstractmethod
from typing import Literal, Iterable, Dict, Callable

from pydantic_settings import BaseSettings

from bus.bus_types import BusProto
from torch.nn import Module
from transformers import Trainer, TrainingArguments


class DatasetWrapperProto(ABC):

    @abstractmethod
    def wrap_dataset(self, data: Iterable[object] | array.ArrayType):
        pass

    @abstractmethod
    def wrap_dataset_cvat(self, data):
        pass


EventTypeLiteral = Literal["filter", "al", "files"]


class AnnotationServiceClientProto(ABC):

    def __init__(self, settings: BaseSettings):
        self.settings = settings

    def download_data(self, task_id, mode):
        """Получает данные из задачи"""

    def send_data_to_service(self, data, config, *args, **kwargs):
        """Метод отправки данных в сервис разметки"""


class EventWrapperProto(ABC):

    @abstractmethod
    def detect_event_type(self, data, *args, **kwargs) -> EventTypeLiteral:
        pass


class HandlerProto(ABC):

    def __init__(self, annotation_service: AnnotationServiceClientProto):
        self.annotation_service = annotation_service

    @abstractmethod
    def handle_filter(self, task_id):
        pass

    @abstractmethod
    def handle_al(self, task_id):
        pass

    @abstractmethod
    def handle_files(self, task_id):
        pass


class FilterServiceProto(ABC):

    def __init__(self,
                 service: BusProto,
                 route: str,
                 filter: Callable,
                 annotation_service: AnnotationServiceClientProto
                 ):
        self.service = service
        self.route = route
        self.filter = filter
        self.annotation_service = annotation_service

    def filtering(self, data):
        """Классифицирует данные для создания задачи"""

    def send_data_to_annotation_client(self, data, task_config):
        """Отправляет данные в cvat для постановки задачи"""


class AnnotationServiceHookProto(ABC):

    def __init__(self,
                 bus: BusProto,
                 handlers,  # : Dict[EventTypeLiteral, HandlerProto]
                 event_wrapper: EventWrapperProto,
                 filter: FilterServiceProto
                 ):
        self.bus = bus
        self.handlers = handlers
        self.event_wrapper = event_wrapper
        self.filter = filter

    def listen(self, request):
        """Слушает что приходит из сервиса разметки"""

    def send_data(self, route, images, annotation=None):
        """Отправляет разметку на маршрут"""


class TrainerServiceProto(ABC):
    def __init__(self,
                 service: BusProto,
                 route: str,
                 trainable: Module,
                 annot_client_proto: AnnotationServiceClientProto,
                 trainer_class: Trainer,
                 training_arguments: TrainingArguments,
                 dataset_wrapper: DatasetWrapperProto
                 ):
        self.service = service
        self.route = route
        self.trainable = trainable
        self.annot_client_proto = annot_client_proto
        self.trainer_class = trainer_class
        self.training_arguments = training_arguments
        self.dataset_wrapper = dataset_wrapper

    def train_proces(self, dataset):
        """Метод обучения на данных"""

    def send_to_bus(self, train_data):
        """Метод отправки данных в шину для AL"""

    async def listen(self):
        """Метод опроса шины на предмет наличия датасета"""

    def create_new_annotation_task(self, file_hashes, config):
        """Метод создает новую задачу в сервисе обучения"""


class ALServiceProto(ABC):

    def __init__(self,
                 service: BusProto,
                 route: str,
                 annot_client_proto: AnnotationServiceClientProto
                 ):
        self.service = service
        self.route = route
        self.annot_client_proto = annot_client_proto

    def train(self):
        """Метод обучения на данных"""

    def send_to_annotation_serv(self):
        """Метод отправки в сервис разметки"""

    def listen(self, request):
        """Метод прослушивания шины"""
