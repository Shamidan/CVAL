import copy
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from scripts.classification.util import read_image

latent_dim = 2
margin = 1


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet34(weights=None)
        # self.resnet = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, latent_dim),
        )
        self.resnet.apply(init_weights)
        self.fc.apply(init_weights)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, anchor, positive, negative):
        # get two images' features
        output1 = self.forward_once(anchor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)

        return output1, output2, output3


class MATCHER(Dataset):
    def __init__(self, path_image, num_labels, dict_id, len_ds=-1):
        super(MATCHER, self).__init__()
        self.grouped_examples = None
        path_to_numpy = os.path.join(path_image, 'numpy_siam')
        if not os.path.exists(path_to_numpy):
            os.mkdir(path_to_numpy)
        labeled_files, labeled_class = self.read_labels(dict_id)
        dict_img = self.read_images(path_image, labeled_files)
        data_list = []
        for file in labeled_files:
            data_list.append(dict_img[file])
        self.data = np.array(data_list)
        self.targets = np.array([int(x) for x in labeled_class])
        self.lends = len_ds
        self.N_class = num_labels
        self.group_examples()

    @staticmethod
    def read_images(path_image, labeled_files):
        files = os.listdir(path_image)
        for del_dir in ['numpy_siam', 'numpy']:
            if del_dir in files:
                files.remove(del_dir)
        dict_img = {}
        for file in files:
            if file not in labeled_files:
                continue
            path_to_numpy = os.path.join(path_image, 'numpy_siam', file + '.npy')
            if os.path.exists(path_to_numpy):
                dict_img[file] = np.load(path_to_numpy)
            else:
                np_img = read_image(path_image, file)
                np.save(path_to_numpy, np_img)

                dict_img[file] = np_img
        return dict_img

    @staticmethod
    def read_labels(dict_id):
        labeled_data = []
        labeled_class = []
        for k, v in dict_id.items():
            labeled_data.append(k)
            labeled_class.append(v[1])

        return labeled_data, labeled_class

    def group_examples(self):
        np_arr = self.targets

        self.grouped_examples = {}
        for i in range(0, self.N_class):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def __len__(self):
        if self.lends != -1:
            return self.lends
        return self.data.shape[0]

    def __getitem__(self, index):
        selected_class = random.randint(0, self.N_class - 1)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        anchor = self.grouped_examples[selected_class][random_index_1]
        anchor = self.data[anchor]

        random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        while random_index_2 == random_index_1:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        positive = self.grouped_examples[selected_class][random_index_2]
        positive = self.data[positive]

        other_selected_class = random.randint(0, self.N_class - 1)
        while other_selected_class == selected_class:
            other_selected_class = random.randint(0, self.N_class - 1)
        random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)
        negative = self.grouped_examples[other_selected_class][random_index_2]
        negative = self.data[negative]

        tran = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])
        anchor = tran(anchor)
        positive = tran(positive)
        negative = tran(negative)

        return anchor, positive, negative, index


def train(model, device, train_loader, optimizer):
    model.train()

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    # nnn = 0
    for batch_idx, (anchor, positive, negative, ind) in enumerate(train_loader):
        if batch_idx == 100:
            break
        # if batch_idx == 0:
        #     print(ind)
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        # nnn += len(anchor)
        # if batch_idx % 50 == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, nnn, len(train_loader.dataset),
        #            100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    with torch.no_grad():
        for (anchor, positive, negative, ind) in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            test_loss += criterion(anchor_emb, positive_emb, negative_emb).sum().item()  # sum up batch loss

            correct = calc_metric(anchor_emb, correct, negative_emb, positive_emb)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def val(model, device, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for (anchor, positive, negative, ind) in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)

            correct = calc_metric(anchor_emb, correct, negative_emb, positive_emb)

    return 100. * correct / len(test_loader.dataset)


def calc_metric(anchor_emb, correct, negative_emb, positive_emb):
    dist1 = (anchor_emb - positive_emb).pow(2).sum(1).sqrt()
    dist2 = (anchor_emb - negative_emb).pow(2).sum(1).sqrt()
    pred = torch.where(dist2 > dist1, 1, 0)
    correct += pred.sum().item()
    return correct


def main(path_to_dir, device, num_labels, dict_id, epochs, validation,
         path_to_dataset_img_val, dict_id_val):
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 16}
    # cuda_kwargs = {'num_workers': 1,
    #                'pin_memory': True,
    #                'shuffle': True}
    cuda_kwargs = {'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    # print('datasets')
    train_dataset = MATCHER(path_to_dir, num_labels, dict_id, len(dict_id) * 30)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    if validation:
        best_acc = 0
        best_model = None
        val_dataset = MATCHER(path_to_dataset_img_val, num_labels, dict_id_val, len(dict_id_val) * 10)
        val_loader = torch.utils.data.DataLoader(val_dataset, **train_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    # print(epochs)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        if validation:
            acc = val(model, device, val_loader)
            # print(acc)
            if best_acc < acc:
                best_model = copy.deepcopy(model)
                best_acc = acc
        scheduler.step()
    # print(model)
    if validation:
        return best_model
    else:
        return model


if __name__ == '__main__':
    # main()
    pass