import random
import os
import multiprocessing
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform
from sklearn.mixture import GaussianMixture
import scripts.config as src_config
from scripts.classification.util import read_image


class DatasetSiam(Dataset):

    def __init__(self, path_image, num_labels, labeled_files):
        super(DatasetSiam, self).__init__()
        path_to_numpy = os.path.join(path_image, 'numpy_siam')
        if not os.path.exists(path_to_numpy):
            os.mkdir(path_to_numpy)
        self.N_class = num_labels
        self.dir_images = path_image
        self.labeled_files = labeled_files
        self.tran = transform.Compose([transform.ToPILImage(),
                                       transform.RandomHorizontalFlip(),
                                       transform.ToTensor()])

        jobs = [multiprocessing.Process(target=self.worker,
                                        args=(fn,),
                                        )
                for fn in self.labeled_files]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, index):
        anchor = self.read_images(self.labeled_files[index])
        anchor = self.tran(anchor)

        return anchor

    def worker(self, file):
        path_to_numpy = os.path.join(self.dir_images, 'numpy_siam', file + '.npy')
        if not os.path.exists(path_to_numpy):
            np_img = read_image(self.dir_images, file)
            np.save(path_to_numpy, np_img)

    def read_images(self, file):
        path_to_numpy = os.path.join(self.dir_images, 'numpy_siam', file + '.npy')
        out = np.load(path_to_numpy)
        return out

    @staticmethod
    def read_labels(dict_id):
        labeled_data = []
        labeled_class = []
        for k, v in dict_id.items():
            labeled_data.append(k)
            labeled_class.append(v[1])

        return labeled_data, labeled_class


def cluster(model, device, path_to_dir, num_labels, training_data, unlabeled_data, num_sample):
    model.eval()
    # print('eval')
    train_dataset = DatasetSiam(path_to_dir, num_labels, training_data)
    train_loader = DataLoader(train_dataset, batch_size=src_config.BATCH_SIZE_TRAIN_SIAM)

    data = []
    with torch.no_grad():
        for images in train_loader:
            images = images.to(device)
            image_emb = model.forward_once(images)
            data.append(image_emb.detach().cpu().numpy())

    conk_data = np.concatenate(data)

    mix = GaussianMixture(n_components=num_labels, covariance_type='full')
    mix.fit(conk_data)
    # print('test')
    test_dataset = DatasetSiam(path_to_dir, num_labels, unlabeled_data)
    test_loader = DataLoader(test_dataset, batch_size=src_config.BATCH_SIZE_TEST_SIAM, shuffle=False)
    y_out = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            image_emb = model.forward_once(images)
            x2 = image_emb.detach().cpu().numpy()
            y_ = mix.predict(x2)
            y_out.append(y_.tolist())
    p = []
    y_ = np.concatenate(y_out)
    for kk in range(num_labels):
        index = np.where(y_ == kk)[0]
        p.append(len(index))
    p = [x/sum(p) for x in p]
    p = [(1-x)/sum(p) for x in p]

    out = []
    for kk in range(num_labels):
        index = np.where(y_ == kk)[0].tolist()
        index_r = random.sample(index, k=min(int(num_sample * p[kk]), len(index)))
        out = out + [unlabeled_data[x] for x in index_r]
    if num_sample - len(out) > 0:
        res = list(set(unlabeled_data) - set(out))
        out = out + random.sample(res, k=num_sample - len(out))
    return out
