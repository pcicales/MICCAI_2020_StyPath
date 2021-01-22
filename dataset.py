import PIL
from torchvision import transforms
import h5py
import torch
import numpy as np
from config import options
from utils.logger_utils import Logger

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, mode='train', data_len=None):

        self.mode = mode
        if mode == 'train':
            print('Loading the training data...')
            train_idx = list(range(5))
            train_idx.remove(options.loos)
            images = np.zeros((0, options.img_c, options.img_h, options.img_w))
            labels = np.array([])
            if options.GAN == 1:
                h5f_GAN = h5py.File('<your_h5_stypath_file>', 'r')
                x_GAN = np.transpose(h5f_GAN['x'][:], [0, 3, 1, 2]).astype(int)
                y_GAN = h5f_GAN['y'][:].astype(int)
                name_GAN = h5f_GAN['name'][:]
                style_GAN = h5f_GAN['style'][:]
                hf5_test = h5py.File('<your_h5_test_file>'
                            .format(options.loos), 'r')
                name_test = hf5_test['name'][:]
            for idx in train_idx:
                h5f = h5py.File('<your_h5_train_file>'
                                .format(idx), 'r')
                x = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)
                y = h5f['y'][:].astype(int)
                name = h5f['name'][:]
                if options.GAN == 1:
                    for i in range(0,len(name_GAN)):
                        if (name_GAN[i] in name) and (style_GAN[i] not in name_test):
                            x = np.concatenate((x,[x_GAN[i]]), axis=0)
                            y = np.concatenate((y,[y_GAN[i]]), axis=0)
                images = np.concatenate((images, x), axis=0)
                labels = np.append(labels, y)
                h5f.close()
            print('Training Data Label Counts:')
            idx0 = len(np.where(labels == 0)[0])
            print('Train label 0 (noABMR) image count: ' + str(idx0))
            idx1 = len(np.where(labels == 1)[0])
            print('Train label 1 (ABMR) image count: ' + str(idx1))
            self.images = images
            self.labels = labels

        elif mode == 'test':
            print('Loading the test data...')
            h5f = h5py.File('<your_h5_train_file>'
                            .format(options.loos), 'r')
            self.images = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)[:data_len]
            self.labels = h5f['y'][:].astype(int)[:data_len]
            h5f.close()

            print('Testing Data Label Counts:')
            idx0 = len(np.where(self.labels == 0)[0])
            print('Test label 0 (noABMR) image count: ' + str(idx0))
            idx1 = len(np.where(self.labels == 1)[0])
            print('Test label 1 (ABMR) image count: ' + str(idx1))

    def __getitem__(self, index):

        # img = torch.tensor(self.images[index]).div(255.).float()

        img = torch.tensor(self.images[index]).float()
        img = (img - img.min()) / (img.max() - img.min())
        target = torch.tensor(self.labels[index]).long()

        if self.mode == 'train':
            # normalization & augmentation
            img = transforms.ToPILImage()(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            # img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)
            img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
            img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)

        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.labels)
