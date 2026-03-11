import io
from typing import Dict, List, Any, Iterable
from bus.bus_types import BusProto
import requests


class BusHTTP(BusProto):

    def __init__(self, url: str):
        self.url = url + '/api'

    def upload_files(self, file: io.BytesIO) -> Any:
        files = {'file': ('upload_files.zip', file, 'application/zip')}
        response = requests.post(f'{self.url}/file/items', files=files)
        return response

    def init_annotation(self, route_name: str, file_hashes: List[str]) -> Any:
        params = {
            "route_name": route_name,
        }

        response = requests.post(f'{self.url}/file/annotation/items/initial', params=params, json=file_hashes)
        return response

    def upload_annotation(self, route_name: str, new_annotation_zip: io.BytesIO) -> Any:
        params = {
            "route_name": route_name,
        }
        files = {'new_markup_zip': ('upload_annotation.zip', new_annotation_zip, 'application/zip')}

        response = requests.post(f'{self.url}/file/annotation/items', files=files, params=params)
        return response

    def file_exist(self, file_hash: str) -> bool:
        response = requests.get(f'{self.url}/file/{file_hash}/existence')
        if response.status_code == 200:
            return True
        return False

    def get_annotated_files_hash(self, route_name: str) -> Iterable[str]:
        response = requests.get(f'{self.url}/file/items/{route_name}/annotated/items/hash')
        if response.status_code == 200:
            return response.json()
        return None

    def get_unannotated_files_hash(self, route_name: str) -> Iterable[str]:
        response = requests.get(f'{self.url}/file/items/{route_name}/not-annotated/items/hash')
        return response.json()

    def get_annotation_by_file_hash(self, file_hash: str) -> Dict:
        response = requests.get(f'{self.url}/file/{file_hash}/annotation')

        return response.json()

    def get_file_by_hash(self, file_hash: str) -> io.BytesIO | int:
        response = requests.get(f'{self.url}/file/{file_hash}/', stream=True)
        if response.status_code == 200:
            file_content = io.BytesIO()
            for chunk in response.iter_content():
                if chunk:
                    file_content.write(chunk)

            return file_content
        return response.status_code

    def delete_route_annotations(self, route_name: str) -> Any:
        params = {"route_name": route_name}
        response = requests.delete(f'{self.url}/annotation/items', params=params)
        return response

    def clean_bus(self) -> Dict:
        response = requests.delete(f'{self.url}/api/bus')
        return response


bus = BusHTTP('http://localhost:8001/api')
