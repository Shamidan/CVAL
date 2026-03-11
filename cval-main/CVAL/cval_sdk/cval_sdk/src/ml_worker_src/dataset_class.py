# import array
# from typing import Iterable
# import numpy as np
# from PIL import Image
# from datasets import Dataset
# from torch import nn
# from transformers import TrainingArguments
from abc_types import DatasetWrapperProto
from ml_worker_src.models.exp_3 import CVATDataset
import torch
import numpy as np
from PIL import Image, ImageDraw
from datasets import Dataset
from transformers import TrainingArguments
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, processed_data, target_size=(224, 224)):

        self.data = processed_data
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        # Приведение изображения к ожидаемому формату
        if isinstance(image, dict):
            # Если image — это dict, преобразуем его в изображение
            image = Image.fromarray(image["data"])

        if isinstance(mask, dict):
            # Если mask — это dict, преобразуем его в изображение
            mask = Image.fromarray(mask["data"])  # Предполагается, что в "data" лежат пиксели маски

        # Приводим изображение и маску к размеру 224x224
        image = F.resize(image, size=self.target_size)
        mask = F.resize(mask, size=self.target_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # Преобразуем в тензоры
        image = F.to_tensor(image)  # [C, H, W]
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # [H, W]

        return image, mask


def parse_cvat_json(json_data):
    parsed_annotations = {}
    for shape in json_data[0]["shapes"]:
        frame = shape["frame"]
        label_id = shape["label_id"]
        points = shape["points"]
        if frame not in parsed_annotations:
            parsed_annotations[frame] = {"boxes": [], "labels": []}
        parsed_annotations[frame]["boxes"].append(points)
        parsed_annotations[frame]["labels"].append(label_id)
    return parsed_annotations


class DatasetWrapper(DatasetWrapperProto):

    def wrap_dataset(self, dataset: Dataset):

        data = {
            "x": [dataset[i]["x"] for i in range(len(dataset))],
            "labels": [dataset[i]["labels"] for i in range(len(dataset))],
        }
        return Dataset.from_dict(data)

    def wrap_dataset_cvat(self, data):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        images = data['files']
        raw_labels = data['labels']

        all_labels = set()
        annotations = []

        for label_data in raw_labels:
            for file_hash, annotation in label_data.items():
                all_labels.update(annotation['labels'])
                annotations.append({
                    "boxes": annotation["boxes"],
                    "labels": annotation["labels"]
                })

        class_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

        dataset = CVATDataset(annotations, images, class_to_idx, transform=transform)

        return dataset

    def wrap_segmentation_dataset(self, data):

        images = data["files"]
        labels = data["labels"]
        processed_data = []

        for image_stream, label_dict in zip(images, labels):
            try:
                image = Image.open(image_stream).convert("RGB")
            except Exception as e:
                print(f"Ошибка при загрузке изображения: {e}")
                continue

            image_width, image_height = image.size

            for hash_key, annotation in label_dict.items():
                yolo_annotations = []
                for box, label in zip(annotation["boxes"], annotation["labels"]):
                    normalized_polygon = []
                    for i in range(0, len(box), 2):
                        x = box[i] / image_width
                        y = box[i + 1] / image_height

                        # Обрезаем точки, выходящие за пределы [0, 1]
                        x = min(max(x, 0), 1)
                        y = min(max(y, 0), 1)

                        normalized_polygon.extend([x, y])

                    if label == 43:
                        label = 0
                    elif label == 27:
                        label = 1

                    if normalized_polygon:
                        yolo_annotations.append((label, normalized_polygon))

                if yolo_annotations:
                    processed_data.append((image, yolo_annotations))

        return processed_data
    # return YOLODataset(processed_data, target_size=(224, 224))


training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
)


def generate_segmentation_mask(image_size, annotations):
    mask = Image.new("L", image_size, 0)  # Создаем пустую черно-белую маску
    draw = ImageDraw.Draw(mask)

    for _, x_center, y_center, width, height in annotations:
        # Переводим координаты из нормализованных в пиксельные
        x_min = int((x_center - width / 2) * image_size[0])
        x_max = int((x_center + width / 2) * image_size[0])
        y_min = int((y_center - height / 2) * image_size[1])
        y_max = int((y_center + height / 2) * image_size[1])

        # Рисуем прямоугольник (или полигон) на маске
        draw.rectangle([x_min, y_min, x_max, y_max], outline=1, fill=1)

    return np.array(mask)  # Конвертируем в массив


import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize


class SegmentationDataset(Dataset):
    def __init__(self, processed_data, image_size=(224, 224)):
        self.data = processed_data
        self.image_size = image_size
        self.to_tensor = ToTensor()
        self.resize = Resize(image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, annotations = self.data[idx]

        # Генерируем маску сегментации
        mask = generate_segmentation_mask(image.size, annotations)

        # Приводим изображение и маску к нужному размеру
        image = self.resize(image)
        mask = Image.fromarray(mask)
        mask = self.resize(mask)

        # Преобразуем в тензоры
        image = self.to_tensor(image)  # [C, H, W]
        mask = torch.tensor(np.array(mask), dtype=torch.float32)

        return {
            "images": image,  # Обозначение для входных данных
            "labels": mask  # Обозначение для маски
        }



