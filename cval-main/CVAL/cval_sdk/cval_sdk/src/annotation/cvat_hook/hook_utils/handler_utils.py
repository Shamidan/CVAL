import hashlib
import io
import json
import zipfile

import os

from cvat_settings import CVATSettings

cvat_settings = CVATSettings()


def create_zip(images_data=None, annotation_data=None, zip_name="output.zip", files_data=None):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        if images_data:
            frame_id = 0
            for image_data in images_data:
                image_name = f'frame_{frame_id}.jpg'
                zipf.writestr(os.path.join('files_data', image_name), image_data.getvalue())
                print(f"Добавлено изображение в архив: {image_name}")
                frame_id += 1

        if annotation_data:
            anat_json = {}
            for key, val in annotation_data.items():
                anat_json[key] = val

            annotation_json = json.dumps(anat_json, indent=4)

            zipf.writestr(os.path.join('labels_data', 'annotation.json'), annotation_json)
            print(f"Добавлен файл аннотаций в архив: annotation.json")
        if files_data:
            json_files = json.dumps(files_data, indent=4)
            zipf.writestr(os.path.join('json_data', 'proces_result.json'), json_files)

    print(f"Архив '{zip_name}' успешно создан!")
    return zip_name
