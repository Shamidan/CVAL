import inspect
from torch import nn
from ultralytics import YOLO
import torch.nn.functional as F


class YOLOv8ForClassification(nn.Module):
    def __init__(self, model_path, num_classes):
        super(YOLOv8ForClassification, self).__init__()
        # Загружаем YOLO модель
        yolo = YOLO(model_path)

        # Получаем внутреннюю модель из YOLO
        self.model = yolo.model.model  # Только слои модели

        # Доступ к классификационному слою
        if hasattr(self.model[-1], "linear"):
            in_features = self.model[-1].linear.in_features
            self.model[-1].linear = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Could not find a compatible classification layer in the YOLO model.")

    def forward(self, pixel_values, labels=None):
        # Прямой вызов внутренней модели
        outputs = self.model(pixel_values)

        # Если модель возвращает кортеж, извлекаем логиты
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}

from transformers import Trainer

class YOLOTrainer(Trainer):
    def __init__(self, data_config, *args, **kwargs):
        """
        Кастомный Trainer для использования YOLO API.
        Args:
            yolo_model: Объект модели YOLO.
            data_config: Конфигурация данных YOLO (путь к данным, классы и т.д.).
            args, kwargs: Аргументы для базового Trainer.
        """
        super().__init__(*args, **kwargs)
        self.yolo_model = self.model
        self.data_config = data_config

    def train(self):
        """
        Переопределяем метод train, чтобы использовать YOLO API.
        """
        print("Starting YOLO training...")
        self.yolo_model.train(data=self.data_config, epochs=2)
        print("YOLO training completed.")
# model = YOLO('yolov8s-cls.pt')
#
# # Вывод структуры модели
# print(model.model)