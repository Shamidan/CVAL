import os
import zipfile
from pathlib import Path


#
def create_zip(zip_name: str, images: dict, files: dict, json_files: dict):
    """Функция для создания zip архива"""

    zip_path = Path(f"{zip_name}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Добавляем изображения
        for image_name, image_data in images.items():
            image_path = f"images_data/{image_name}"
            zipf.writestr(image_path, image_data)

        # Добавляем текстовые файлы
        for file_name, file_data in files.items():
            file_path = f"files_data/{file_name}"
            zipf.writestr(file_path, file_data)

        # Добавляем JSON файлы
        for json_name, json_data in json_files.items():
            json_path = f"json_data/{json_name}"
            zipf.writestr(json_path, json_data)

    return zip_path


def extract_zip(zip_path: str):
    """Функция для извлечения содержимого zip архива"""
    files = {}
    labels = {}
    json_data = {}
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for file in zipf.namelist():
            if file.startswith("files_data/"):
                files[os.path.basename(file)] = zipf.read(file)
            elif file.startswith("lables_data/"):
                labels[os.path.basename(file)] = zipf.read(file)
            elif file.startswith("json_data/"):
                json_data[os.path.basename(file)] = zipf.read(file)
    return files, labels, json_data

