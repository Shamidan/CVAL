import io
from typing import List
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18


def cls_funk_2(images: List[io.BytesIO]) -> List[io.BytesIO]:
    model = resnet18(pretrained=True)
    model.eval()

    cat_classes = list(range(281, 286))
    dog_classes = list(range(151, 269))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    filtered_images = []

    for img_bytes in images:
        try:
            image = Image.open(img_bytes).convert("RGB")
        except Exception as e:
            print(f"{e}")
            continue

        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        predicted_class = torch.argmax(probabilities).item()

        if predicted_class in cat_classes or predicted_class in dog_classes:
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="JPEG")
            output_buffer.seek(0)
            filtered_images.append(output_buffer)

    return filtered_images

