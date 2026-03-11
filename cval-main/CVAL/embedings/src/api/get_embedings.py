import glob
import os
import random
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Tuple, Union
from contextlib import suppress
from api.models.detection import FramePrediction, BBoxScores
from api.models.embedding import FrameEmbeddingModel, EmbeddingModel
import PIL.Image as Image
import albumentations as alb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch.optim as optim
import torchvision
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights


def normalize_pattern(_p: str) -> str:
    return _p.replace('*', '')


def get_before_last_dot(_s: str) -> str:
    if match := re.search(r'(.*)\.', _s):
        return match.group(1)
    return _s


def get_triplet_loss(*args, **kwargs) -> nn.TripletMarginWithDistanceLoss:
    return nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - t_func.cosine_similarity(x, y))


def log_train(corrects, sum_loss, train_loader, epoch):
    print(
        'Train Epoch: {} \tLoss: {:.6f}, true: {}/{} ({:.3f}%)'.format(
            epoch,
            sum_loss,
            corrects,
            len(train_loader.dataset),
            100. * corrects / len(train_loader.dataset)
        )
    )


def get_np_anchor(
        id_file: str,
        numpy: np.array,
        images: list[str],
        sz: int = 224,
):
    if os.path.exists(numpy[id_file]):
        np_anchor = np.load(numpy[id_file])
    else:
        pil = Image.open(images[id_file]).convert('RGB')
        pil = pil.resize((sz, sz))
        np.save(numpy[id_file], pil)
        np_anchor = np.array(pil)
    return np_anchor


def criterion(a, p, n):
    output = get_triplet_loss()(a, p, n)
    return output


def calc_metric(
        anchor_emb: np.array,
        negative_emb: np.array,
        positive_emb: np.array,
):
    similarity1 = t_func.cosine_similarity(anchor_emb, positive_emb)
    similarity2 = t_func.cosine_similarity(anchor_emb, negative_emb)
    prediction = torch.where(similarity1 > similarity2, 1, 0)
    correct = prediction.sum().item()
    return correct


def save_bbox_for_cluster(
        path_to_bboxes: Union[Path, str, os.PathLike],
        path_to_images: Union[Path, str, os.PathLike],
        pattern: str = '*.jpg',
) -> Path:
    with suppress(Exception):
        shutil.rmtree(Path(path_to_bboxes) / 'crops')
    os.makedirs(Path(path_to_bboxes) / 'crops', exist_ok=True)
    for path_to_bbox in map(
            lambda x: Path(path_to_bboxes) / x,
            os.listdir(path_to_bboxes)
    ):
        if os.path.exists(path_to_bbox) and Path(path_to_bbox).is_file():
            with open(path_to_bbox) as f:
                bbox = f.readlines()
            image = Image.open(
                Path(path_to_images) /
                (
                        get_before_last_dot(path_to_bbox.name) +
                        normalize_pattern(pattern)
                )
            )
            for i, line in enumerate(bbox):
                if len(line.strip()) == 0:
                    continue
                arr = line.split(' ')
                classes = arr[0]
                xc, yc = float(arr[1]), float(arr[2])
                w, h = float(arr[3]), float(arr[4])
                x1, x2 = max(0, int((xc - w / 2) * image.width)), min(image.width, int((xc + w / 2) * image.width))
                y1, y2 = max(0, int((yc - h / 2) * image.height)), min(image.height, int((yc + h / 2) * image.height))
                image.crop((x1, y1, x2, y2)).save(
                    os.path.join(
                        str(path_to_bboxes),
                        'crops',
                        f'{get_before_last_dot(path_to_bbox.name)}__{i}__{classes}_.jpg',
                    )
                )
    return Path(path_to_bboxes) / 'crops'


class CustomDataset(Dataset):
    def __init__(
            self,
            path_to_crops: Union[Path, str, os.PathLike] = '',
            pattern: str = '*.jpg',
    ):
        super(CustomDataset, self).__init__()
        allfiles = glob.glob(os.path.join(path_to_crops, pattern))
        self.images = allfiles
        self.numpy = [
            x.replace(normalize_pattern(pattern), '.npy') for x in allfiles
        ]
        self.len_images = len(self.images)

    def __len__(self):
        return self.len_images

    def save_pillow(self, id_file: str):
        return get_np_anchor(id_file, self.numpy, self.images)

    def __getitem__(self, ind_anchor: str):
        np_anchor = self.save_pillow(ind_anchor)
        tran = transforms.Compose([transforms.ToTensor()])
        anchor = tran(np_anchor)
        return anchor, self.images[ind_anchor]


class Matcher(Dataset):
    def __init__(
            self,
            path_to_crops: Union[Path, str, os.PathLike],
            pattern: str = '*.jpg',
    ):
        super(Matcher, self).__init__()
        allfiles = sorted(glob.glob(os.path.join(path_to_crops, pattern)))
        self.images = allfiles
        self.images_class = [os.path.basename(x).split('__')[-2] for x in allfiles]
        self.all_class = list(set(self.images_class))
        print(f'True classes {Counter(self.images_class)}')
        dict_cls_files = {}
        for cls in self.all_class:
            dict_cls_files[cls] = [x for x, y in zip(self.images, self.images_class) if y == cls]

        self.dict_cls_files = dict_cls_files
        self.numpy = [x.replace(normalize_pattern(pattern), '.npy') for x in allfiles]
        self.len_images = len(self.images)

        self.tran1 = alb.Compose(
            [
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
            ]
        )
        self.tran2 = alb.Compose(
            [alb.ToGray(p=1)],
        )
        self.tran3 = alb.Compose(
            [alb.RGBShift(p=1)],
        )
        self.tran4 = alb.Compose(
            [alb.RandomRotate90(p=1)],
        )
        self.tran5 = alb.Compose(
            [
                alb.CoarseDropout(
                    min_holes=1,
                    max_holes=1,
                    max_height=20,
                    max_width=20,
                    p=1,
                )
            ],
        )
        self.tran6 = alb.Compose(
            [
                alb.ShiftScaleRotate(
                    p=1.0,
                    shift_limit_x=(-0.06, 0.06),
                    shift_limit_y=(-0.06, 0.06),
                    scale_limit=(-0.1, 0.1),
                    rotate_limit=(-36, 36),
                    interpolation=1,
                    border_mode=1,
                    value=(0, 0, 0),
                    mask_value=None,
                    rotate_method='largest_box'),
            ]
        )

    def __len__(self):
        return self.len_images

    def save_pillow(self, id_file: int):
        return get_np_anchor(id_file, self.numpy, self.images)

    def __getitem__(self, ind_anchor: int):
        random.seed(None)
        class_anchore = self.images_class[ind_anchor]
        neg_class = random.choice(list(set(self.all_class) - {class_anchore}))
        neg_image = random.choice(self.dict_cls_files[neg_class])
        ind_neg = self.images.index(neg_image)

        images_this_class = self.dict_cls_files[class_anchore]
        if len(images_this_class) == 1:
            ind_pos_another = ind_anchor
        else:
            while True:
                p_image = random.choice(self.dict_cls_files[class_anchore])
                ind_pos_another = self.images.index(p_image)
                if ind_pos_another != ind_anchor:
                    break

        np_anchor = self.save_pillow(ind_anchor)
        np_pos_another = self.save_pillow(ind_pos_another)
        np_neg = self.save_pillow(ind_neg)

        trans = random.choice(
            [
                self.tran1,
                self.tran2,
                self.tran3,
                self.tran4,
                self.tran5,
                self.tran6,
            ]
        )
        if random.random() > 0.1:
            np_pos = trans(image=np_anchor)["image"]
        else:
            np_pos = trans(image=np_pos_another)["image"]

        tran = transforms.Compose([transforms.ToTensor()])
        anchor = tran(np_anchor)
        positive = tran(np_pos)
        negative = tran(np_neg)

        return anchor, positive, negative


class SiameseNetwork(nn.Module):
    def __init__(
            self,
            dim: Tuple[int, int],
    ):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, dim),
        )

    def forward_once(self, x: np.array):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, anchor: np.array, positive: np.array, negative: np.array):
        output1 = self.forward_once(anchor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)
        return output1, output2, output3


def train_epoch_siam(
        model: nn.Module,
        dev: object(),
        train_loader: torch.utils.data.DataLoader,
        optimizer: object(),
):
    model.train()
    sum_loss = 0
    corrects = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += loss.item()
        correct = calc_metric(anchor_emb, negative_emb, positive_emb)
        corrects += correct
    return corrects, sum_loss, train_loader


def train_siam(
        path_to_bboxes: Union[Path, str, os.PathLike],
        epochs: int = 30,
        dim: int = 512,
        batch_size: int = 8,
        shuffle: bool = True,
        device: str = 'cuda:0',
):
    train_dataset = Matcher(path_to_bboxes)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    model = SiameseNetwork(dim).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    for epoch in range(1, epochs + 1):
        log_train(*train_epoch_siam(model, device, train_loader, optimizer), epoch)
        torch.save(model.state_dict(), 'end_pool.pt')
        scheduler.step()
    return model


def get_embeddings(
        path_to_crops: Union[Path, str, os.PathLike],
        dim: int = 512,
        device: object() = 'cuda:0',
        batch_size: int = 10,
        shuffle: bool = True,
        pattern: str = '*.jpg',
):
    model = SiameseNetwork(dim)
    model.load_state_dict(
        torch.load('end_pool.pt')
    )
    model.eval()
    model.to(device)
    test_dataset = CustomDataset(
        path_to_crops,
        pattern=pattern,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, )
    embs = generate_embeddings(model, test_loader, device)
    return embs


def generate_embeddings(
        model: SiameseNetwork,
        dataloader: torch.utils.data.DataLoader,
        dev: object()
):
    emb = []
    files = []
    with torch.no_grad():
        for img, frames in dataloader:
            img = img.to(dev)
            emb_batch = model.forward_once(img).detach().cpu().flatten(start_dim=1)
            emb.append(emb_batch)
            files.extend(frames)
    embs = torch.cat(emb, 0)
    embs = normalize(embs)
    embeddings = [
        FrameEmbeddingModel(frame_id=img, embeddings=[EmbeddingModel(embedding_id=str(uuid.uuid4()), embedding=emb)])
        for emb, img in zip(embs.tolist(), map(lambda x: Path(x.split('__')[0]).name, files))
    ]
    return embeddings