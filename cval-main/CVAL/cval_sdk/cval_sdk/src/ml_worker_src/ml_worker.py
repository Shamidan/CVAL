# import asyncio
# import os
# import time
# from datasets import load_dataset
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import numpy as np
# import requests
# import torch
# from torchvision import transforms
# import torch.nn.functional as F
# from PIL import Image
# from transformers import Trainer
# from al_service.al.utils import BBoxScores, FramePrediction, SamplingArguments
# from cvat_settings import CVATSettings, ClientSettings
from annotation.cvat_hook.cvat_hook import generate_file_hash
# from annotation.cvat_hook.hook_utils.hook_clasess import CVATProces
# from ml_worker_src.dataset_class import training_arguments, DatasetWrapper
# from abc_types import TrainerServiceProto
# from bus.http.sync_client import BusHTTP
# from annotation.cvat_hook.hook_utils.handler_utils import create_zip
from ml_worker_src.models.Yolo_test_model import YOLOv8ForClassification, YOLOTrainer
import asyncio
import os
import tempfile
import time
from pprint import pprint
from ultralytics import YOLO
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import requests
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from transformers import Trainer
from al_service.al.utils import BBoxScores, FramePrediction, SamplingArguments
from cvat_settings import CVATSettings, ClientSettings
# from annotation.cvat_hook import generate_file_hash
from annotation.cvat_hook.hook_utils.hook_clasess import CVATProces
from ml_worker_src.dataset_class import training_arguments, DatasetWrapper, \
    generate_segmentation_mask, SegmentationDataset
from abc_types import TrainerServiceProto
from bus.http.sync_client import BusHTTP
from annotation.cvat_hook.hook_utils.handler_utils import create_zip
# from ml_worker_src.models.Yolo_test_model import YOLOv8ForClassification, YOLOv8ForSegmentation
# from ml_worker_src.models.exp_3 import ResNetForClassification

def save_bytesio_to_tempfile(bytes_io_obj):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(bytes_io_obj.getvalue())
    temp_file.close()
    return temp_file.name


def bytesio_to_numpy(bytes_io_obj):
    image = Image.open(bytes_io_obj)
    return np.array(image)


def save_dataset_images_and_annotations(dataset_name, output_dir="images"):
    dataset = load_dataset(dataset_name, split="train")
    os.makedirs(output_dir, exist_ok=True)
    annotations = []
    for idx, item in enumerate(dataset):
        image = item["images"]
        label = item["labels"]
        image_path = os.path.join(output_dir, f"image_{idx}.jpg")
        image.save(image_path)
        annotations.append(label)
    return annotations


def save_segmentation_data(processed_data, base_dir):
    # Директории для изображений и разметки
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    # Создание директорий, если они не существуют
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx, (image, annotations) in enumerate(processed_data):
        # Сохраняем изображение
        image_path = os.path.join(images_dir, f"image_{idx}.jpg")
        image.save(image_path)

        # Сохраняем аннотации в формате YOLO
        label_path = os.path.join(labels_dir, f"image_{idx}.txt")
        with open(label_path, 'w') as label_file:
            for annotation in annotations:
                label = annotation[0]
                polygon = annotation[1]
                # Записываем в формате: class_id x1 y1 x2 y2 ... xn yn
                label_file.write(f"{label} " + " ".join(f"{coord:.6f}" for coord in polygon) + "\n")

    print(f"Данные успешно сохранены в директориях: {images_dir} и {labels_dir}")


def subtract_lists(all_files, data_files):
    result = []
    for item in all_files:
        if item not in data_files:
            result.append(item)
    return result


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {'x': {"img": images, "cls": labels}}


def yolo_collate_fn(batch):
    images = []
    masks = []

    for item in batch:
        image, mask = item
        images.append(image)
        masks.append(mask)

    batch_images = torch.stack(images)  # Собираем изображения в тензор
    batch_masks = torch.stack(masks)  # Собираем маски в тензор

    return {
        "images": batch_images,  # Тензор изображений [B, C, H, W]
        "labels": batch_masks,  # Тензор масок [B, H, W]
    }


def segmentation_collate_fn(batch):
    pixel_values = torch.stack([item["images"] for item in batch])  # [B, C, H, W]
    labels = torch.stack([item["labels"] for item in batch])  # [B, H, W]

    return {
        "images": pixel_values,
        "labels": labels
    }


class TrainerService(TrainerServiceProto):

    def train_proces(self, dataset, segmentation=None, yolo_config=None, data_config=None):

        if segmentation:
            dataset = self.dataset_wrapper.wrap_segmentation_dataset(dataset)
            save_segmentation_data(dataset, r"ml_worker_src/yolo_data")
            dataset = SegmentationDataset(dataset)
        else:
            dataset = self.dataset_wrapper.wrap_dataset_cvat(dataset)

        if data_config:
            self.trainable.overrides['imgsz'] = 128
            trainer = self.trainer_class(
                model=self.trainable,
                data_config=data_config,
                args=self.training_arguments,
            )
            trainer.train()
            model_path = "yolov8n-seg.pt"
            self.yolo_model = YOLO(model_path)  # Загружаем YOLO-модель

            self.yolo_model.save("trained_model.pt")
        else:
            trainer = self.trainer_class(
                model=self.trainable,
                args=self.training_arguments,
                train_dataset=dataset,
                data_collator=segmentation_collate_fn
            )
            trainer.train()
            model_path = "trained_model.pth"
            torch.save(self.trainable.state_dict(), model_path)

        return model_path

    def send_to_bus(self, train_data):
        train_data_hash = generate_file_hash(train_data)
        data_zip = create_zip(files_data=train_data)
        response_upload_files = self.service.upload_files(data_zip)
        response_init_annotation = self.service.init_annotation('al', [train_data_hash])
        return {'response_upload_files': response_upload_files, 'response_init_annotation': response_init_annotation}

    def get_data_from_bus(self, response, task_id=None):

        data_raw = {
            'files': [],
            'labels': []
        }
        task_name = None
        project_id = None

        if not response:
            return False
        file_hashes = response.get('file_hashes')

        for file_hash in file_hashes:

            annotation = self.service.get_annotation_by_file_hash(file_hash)

            if task_name is None:
                markup = annotation.get('markup')
                for key in markup.keys():
                    task_name = markup[key]['task_name']

            if project_id is None:
                markup = annotation.get('markup')
                for key in markup.keys():
                    pprint(markup[key])
                    project_id = markup[key]['project_id']

            file = self.service.get_file_by_hash(file_hash)
            data_raw['files'].append(file)

            if annotation:
                data_raw['labels'].append(annotation.get('markup'))
            else:
                return 'Bad data'

        return data_raw, task_name, project_id

    async def listen(self):
        response = self.service.get_annotated_files_hash(self.route)

        if response:
            dataset, task_name, project_id = self.get_data_from_bus(response)
            return dataset, task_name, project_id
        else:
            return None, None, None

    def _predict_image(self, image_bytes, model):
        image = Image.open(image_bytes).convert("RGB")

        # Преобразования для входа в модель
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        image_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)

        if isinstance(outputs, dict) and "logits" in outputs:
            outputs = outputs["logits"]

        probabilities = F.softmax(outputs, dim=1).squeeze(0)

        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities.tolist()

    def _create_frame_predictions(self, inference_results):
        frame_predictions = []

        for idx, result in enumerate(inference_results):
            file_hash = result['file_hash']
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            bbox_score = BBoxScores(
                category_id=predicted_class,
                score=confidence,
                probabilities=probabilities
            )

            frame_prediction = FramePrediction(
                frame_id=file_hash,
                predictions=[bbox_score]
            )

            frame_predictions.append(frame_prediction)

        return frame_predictions

    def _batch_inference(self, unmarked_frames, model):
        results = []
        for idx, image_bytes in enumerate(unmarked_frames):
            np_image = save_bytesio_to_tempfile(image_bytes)
            predicted_class, confidence, probabilities = self._predict_image(np_image, model)
            results.append({
                "file_hash": generate_file_hash(image_bytes),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities
            })
        return results

    def get_unannotated_files(self, data):
        response = self.service.get_unannotated_files_hash('filter')
        all_hashes = response.get('file_hashes')
        if all_hashes is None:
            return None
        annotated_hashes = []
        for item in data:
            hash_key = generate_file_hash(item)
            annotated_hashes.append(hash_key)

        unannotated_hashes = subtract_lists(all_hashes, annotated_hashes)

        unannotated_files = []
        for file_hash in unannotated_hashes:
            if file_hash in annotated_hashes:
                continue
            file = self.service.get_file_by_hash(file_hash)
            unannotated_files.append(file)
        return unannotated_files

    def _batch_inference_yolo(self, unmarked_frames, yolo_model):
        results = []

        for idx, image_bytes in enumerate(unmarked_frames):
            # Преобразуем байтовый объект в изображение (NumPy)
            np_image = save_bytesio_to_tempfile(image_bytes)

            # Прогоняем изображение через модель YOLO
            yolo_results = yolo_model.predict(source=np_image, save=False, save_txt=False)

            # Инициализация параметров для вывода
            predicted_class = None
            confidence = None
            probabilities = []

            # Обрабатываем результаты инференса
            for detection in yolo_results:
                bbox = detection.boxes.xyxy.numpy()[0]  # координаты бокса
                conf = detection.boxes.conf.numpy()[0]  # уверенность модели
                cls = int(detection.boxes.cls.numpy()[0])  # класс объекта

                # Присваиваем значения предсказания
                predicted_class = cls
                confidence = conf
                probabilities.append(conf)  # Можно дополнить расширенными вероятностями, если доступны

            # Добавляем результат в итоговый список
            results.append({
                "file_hash": generate_file_hash(image_bytes),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities
            })

        return results

    def _create_frame_predictions_yolo(self, inference_results, num_classes):
        frame_predictions = []

        for result in inference_results:
            file_hash = result['file_hash']
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            probabilities = result['probabilities']  # Вероятности из инференса

            # Проверяем длину вероятностей и заполняем недостающие нулями
            if len(probabilities) < num_classes:
                probabilities = probabilities + [0.0] * (num_classes - len(probabilities))

            bbox_score = BBoxScores(
                category_id=predicted_class,
                score=confidence,
                probabilities=probabilities  # Исправленные вероятности
            )

            frame_prediction = FramePrediction(
                frame_id=file_hash,
                predictions=[bbox_score]
            )

            frame_predictions.append(frame_prediction)

        return frame_predictions

    def send_to_al(self, model_path, unmarked_files, config, yolo_mode=None):

        if yolo_mode:
            yolo_model = YOLO(model_path)  # Убедитесь, что model_path указывает на YOLO модель

            # Переключаем модель в режим оценки
            yolo_model.eval()

            # Выполняем инференс для каждого файла
            inference_results = self._batch_inference_yolo(unmarked_files, yolo_model)
            pprint(inference_results)
            # Преобразуем результаты инференса в формат кадров
            frame_predictions = self._create_frame_predictions_yolo(inference_results, 2)
            pprint(frame_predictions)

            # Формируем запрос
            request = SamplingArguments(
                frames=frame_predictions,
                num_of_samples=config['num_of_samples'],
                bbox_selection_policy=config['bbox_selection_policy'],
                selection_strategy=config['selection_strategy'],
                sort_strategy=config['sort_strategy'],
                probs_weights=config['probs_weights']
            )
            pprint(request)

            # Отправляем запрос в зависимости от стратегии
            if config['sort_strategy'] == 'descending':
                response = requests.post('http://cval_embedings:8005/full_process', json=request.dict())
                if response.status_code == 200:
                    return response.json()
                else:
                    return response.status_code

            response = requests.post('http://cval_sampling:8000/api/al', json=request.dict())
            return response.json()

        else:

            state_dict = torch.load(model_path, weights_only=True)

            self.trainable.load_state_dict(state_dict)

            self.trainable.eval()

            inference_results = self._batch_inference(unmarked_files, self.trainable)

            frame_predictions = self._create_frame_predictions(inference_results)

            request = SamplingArguments(
                frames=frame_predictions,
                num_of_samples=config['num_of_samples'],
                bbox_selection_policy=config['bbox_selection_policy'],
                selection_strategy=config['selection_strategy'],
                sort_strategy=config['sort_strategy'],
                probs_weights=config['probs_weights']
            )
            if config['sort_strategy'] == 'descending':
                response = requests.post('http://cval_embedings:8005/full_process', json=request.dict())
                if response.status_code == 200:
                    return response.json()
                else:
                    return response.status_code

            response = requests.post('http://cval_sampling:8000/api/al', json=request.dict())
            return response.json()

    def create_new_annotation_task(self, file_hashes, config):
        data = []
        for i in file_hashes:
            file = self.service.get_file_by_hash(i)
            data.append(file)
        zip_images = create_zip(data, zip_name='from_al.zip')
        self.annot_client_proto.send_data_to_service(zip_images, config)

    def test_model(self, test_directory, true_labels=None):

        # Проверка наличия изображений в тестовой директории
        image_paths = [os.path.join(test_directory, file) for file in os.listdir(test_directory)
                       if file.endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            print("В директории нет изображений!")
            return

        processed_data = []

        # Загрузка обученной модели YOLO
        self.yolo_model = YOLO("trained_model.pt")  # Загружаем весы YOLO
        self.yolo_model.eval()

        for img_path in image_paths:
            # Загрузка изображения
            image = Image.open(img_path).convert("RGB")
            image_width, image_height = image.size

            # Получение предсказаний модели
            results = self.yolo_model.predict(img_path)

            # Извлечение полигонов и классов
            yolo_annotations = []
            for mask, cls in zip(results[0].masks.data.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                # Нормализация координат полигона
                polygons = mask.tolist()
                print(polygons)  # Преобразуем маску в список точек

                normalized_polygon = []
                for x, y in polygons:
                    x /= image_width
                    y /= image_height
                    normalized_polygon.extend([x, y])

                # Добавляем аннотацию
                yolo_annotations.append((int(cls), normalized_polygon))

            processed_data.append((image, yolo_annotations))

        # Опциональная проверка метрик, если даны истинные метки
        if true_labels is not None:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from colorama import Fore, Style

            true_labels = np.array(true_labels)
            predicted_labels = [ann[0] for _, anns in processed_data for ann in anns]

            accuracy = accuracy_score(true_labels, predicted_labels)
            class_report = classification_report(true_labels, predicted_labels, target_names=['Dog', 'Cat'])
            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            print(f"\n{Fore.CYAN}{'=' * 25} Evaluation Metrics {'=' * 25}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Overall Accuracy: {accuracy:.2f}{Style.RESET_ALL}")

            print(f"\n{Fore.GREEN}Classification Report:{Style.RESET_ALL}")
            print(class_report)

            print(f"\n{Fore.MAGENTA}Confusion Matrix:{Style.RESET_ALL}")
            print(conf_matrix)

            print(f"{Fore.CYAN}{'=' * 74}{Style.RESET_ALL}")

        return processed_data

    import torch
    import numpy as np
    from typing import List, Dict, Any

    def yolo_inference(self, model, images: List[np.ndarray], labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        processed_data = []

        for image, label in zip(images, labels):
            # Преобразуем изображение в тензор
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Выполняем инференс модели
            with torch.no_grad():
                predictions = model(image_tensor)[0]

            # Обрабатываем предсказания (координаты и классы)
            boxes = []
            classes = []
            for pred in predictions:
                x_min, y_min, x_max, y_max, conf, class_id = pred[:6].tolist()
                boxes.append([x_min, y_min, x_max, y_max])
                classes.append(int(class_id))

            # Извлекаем данные аннотаций из labels
            annotation = label.get("annotation", "")
            annotation_parts = list(map(float, annotation.split())) if annotation else []
            for i in range(0, len(annotation_parts), 5):
                class_id, x_min, y_min, x_max, y_max = annotation_parts[i:i + 5]
                boxes.append([x_min, y_min, x_max, y_max])
                classes.append(int(class_id))

            # Добавляем обработанные данные в результат
            processed_data.append({
                "image": image,
                "boxes": boxes,
                "labels": classes
            })

        return processed_data


async def main():
    settings = ClientSettings()
    bus = BusHTTP(f'{settings.bus_url}/')
    cvat_settings = CVATSettings()
    cvat = CVATProces(settings=cvat_settings)
    dataset_wrapper = DatasetWrapper()
    # ТУТ МОДЕЛЬ
    # model = YOLOv8ForClassification('yolov8s-cls.pt', num_classes=2)
    # model = YOLOv8ForSegmentation('yolov8n-seg.pt')
    model = YOLO('yolov8n-seg.pt')
    trainer_class = YOLOTrainer  # Trainer
    route = 'files'

    train = TrainerService(bus, route, model, cvat, trainer_class, training_arguments, dataset_wrapper)

    last_data = {'labels': ''}
    iter_num = 1
    dataset_name = settings.dataset_name
    test_dataset_path = 'test_dataset'
    sampling_args = {
        'num_of_samples': 2,
        'bbox_selection_policy': 'sum',
        'selection_strategy': 'cval_custom',  # cval_custom
        'sort_strategy': 'descending',  # ascending
        'probs_weights': [1, 1]
    }
    if dataset_name:
        annotation = save_dataset_images_and_annotations(dataset_name, test_dataset_path)
    while True:
        try:
            data, task_name, project_id = await train.listen()
        except requests.exceptions.ConnectionError:
            continue
        if not data:
            time.sleep(5)
            continue
        elif last_data['labels'] == data['labels']:
            time.sleep(5)
            continue
        task_config = {"name": f"{task_name}_{iter_num}" + '[files]', 'project_id': int(project_id),
                       'image_quality': 100}
        iter_num += 1
        last_data = data

        unannotated_files = train.get_unannotated_files(data['files'])
        if unannotated_files is None:
            print('Все изображения размченны')
            continue

        model_path = train.train_proces(data, segmentation=True, data_config=r"/app/ml_worker_src/yolo_data/data.yaml")
        # if annotation:
        #     train.test_model(test_dataset_path, annotation)
        #
        # else:
        #     train.test_model(test_dataset_path)

        print(f'Все прошло успешно. Путь к модели {model_path}')

        al_values = train.send_to_al(model_path, unannotated_files, sampling_args, yolo_mode=True)
        print(al_values)
        if al_values == 500:
            print('Поломалось')

        train.create_new_annotation_task(al_values, task_config)
        print("=" * 100)


# Запуск main() в асинхронном контексте
if __name__ == "__main__":
    asyncio.run(main())

