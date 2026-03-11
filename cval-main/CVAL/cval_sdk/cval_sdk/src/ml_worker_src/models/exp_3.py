from pprint import pprint

from torch import nn
from torch.utils.data.dataloader import default_collate
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F


# TODO: преобразовать в нормальные тесты


class ResNetForClassification(nn.Module):
    def __init__(self, num_classes):
        super(ResNetForClassification, self).__init__()
        self.resnet = models.resnet18(weights="DEFAULT")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values)  # Прогон через ResNet
        loss = None
        if labels is not None:
            # Убеждаемся, что labels имеют размер [batch_size]
            loss = F.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}


# Парсинг аннотаций
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


# Класс Dataset
class CVATDataset(Dataset):
    def __init__(self, annotations, images, class_to_idx, transform=None):
        self.annotations = annotations
        self.images = images
        self.class_to_idx = class_to_idx
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Загружаем изображение из BytesIO
        image_bytes = self.images[idx]
        image = Image.open(image_bytes).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Загружаем аннотации для текущего изображения
        annotation = self.annotations[idx]
        boxes = torch.tensor(annotation["boxes"], dtype=torch.float32)
        labels = torch.tensor(
            [self.class_to_idx[label] for label in annotation["labels"]],
            dtype=torch.int64
        )

        # Возвращаем данные для классификации
        return {
            "pixel_values": image,
            "labels": labels[0] if len(labels) > 0 else torch.tensor(0, dtype=torch.int64)
        }
