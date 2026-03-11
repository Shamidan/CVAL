"""
Протокол шины
"""

import io
from abc import ABC, abstractmethod
from typing import Any, Iterable, Dict, List


class BusProto(ABC):
    @abstractmethod
    def upload_files(self, file: io.BytesIO) -> Any:
        """
        Метод загрузки файлов
        """

    @abstractmethod
    def init_annotation(self, route_name: str, file_hashes: List[str]) -> Any:
        """
        Метод инициализации аннотаций
        """

    @abstractmethod
    def upload_annotation(self, route_name: str, new_annotation_zip: io.BytesIO) -> Any:
        """
        Метод загрузки аннотаций
        """

    @abstractmethod
    def file_exist(self, file_hash: str) -> bool:
        """
        Метод проверки наличия файла
        """

    @abstractmethod
    def get_annotated_files_hash(self, route_name: str) -> Iterable[str]:
        """
        Метод получения списка хэшей
        файлов для маршрута, у которых есть разметка
        """

    @abstractmethod
    def get_unannotated_files_hash(self, route_name: str) -> Iterable[str]:
        """
        Метод получения списка хэшей
        файлов для маршрута, у которых нет разметки
        """

    @abstractmethod
    def get_annotation_by_file_hash(self, file_hash: str) -> Dict:
        """
        Метод получения разметки файла по хэшу
        """

    @abstractmethod
    def get_file_by_hash(self, file_hash: str) -> io.BytesIO:
        """
        Метод получения файла по хэшу
        """

    @abstractmethod
    def delete_route_annotations(self, route_name: str) -> Any:
        """
        Метод очистки разметки для машрута
        """

    @abstractmethod
    def clean_bus(self) -> Any:
        """
        Метод очистки файлов
        """
