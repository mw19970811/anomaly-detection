from functools import reduce
from util import pprint
import glob, os, tqdm, random
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision

MAX_VALUE = 100


def my_resize(img, ht, wt):
    hi, wi = img.shape[-2:]
    if ht < hi:
        img = img[..., :ht, :]
    else:
        img = torch.nn.functional.pad(img, (0, 0, 0, ht - hi), 'constant', 0)
    if ht < hi:
        img = img[..., :wt, :]
    else:
        img = torch.nn.functional.pad(img, (0, wt - wi), 'constant', 0)
    return img


def get_dataset(root_path, cfg):
    train_transform = torchvision.transforms.Compose([
        RandomRotation(),
        torchvision.transforms.RandomCrop(1024, padding=(0, 0, 512, 512)),
        HistogramEqualization(p=0.),
        RandomColorJitter(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                                                                      [0.5, 0.5, 0.5])])
    train_dataset = AnomalyDataset(root_path, cfg=cfg, mode='train', transform=train_transform)
    val_dataset = AnomalyDataset(root_path, cfg=cfg, mode='val', transform=test_transform)
    # pprint('train dataset length: ',len(train_dataset))
    # pprint('test dataset length: ',len(test_dataset))
    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, cfg, shuffle_train=True, shuffle_test=False):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.mini_batch_size,
                                               num_workers=cfg.nb_workers, shuffle=shuffle_train)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             num_workers=cfg.nb_workers, shuffle=shuffle_test)
    return train_loader, val_loader


class HistogramEqualization(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert isinstance(p, float)
        self.p = p
        self.threshold = 15

    def forward(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img = np.array(img)
        if random.uniform(0, 1) < self.p:
            diff = np.abs(np.mean(np.diff(img.astype(np.float32), axis=1), axis=0))
            diff_special = np.unique(np.argwhere(diff > self.threshold)[:, 0]) + 1
            imgs = np.split(img, diff_special, axis=1)
            oriImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            oriImg_mean, oriImg_std = oriImg.mean(axis=(0, 1)), oriImg.std(axis=(0, 1))
            for i in range(len(imgs)):
                imgs[i] = update(imgs[i], oriImg_mean, oriImg_std)
            img = np.concatenate(imgs, axis=1)
        img = Image.fromarray(img)
        return img


class RandomColorJitter(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.transform = torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)

    def forward(self, img):
        if random.uniform(0, 1) < self.p:
            self.transform(img)
        return img


class RandomRotation(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.uniform(0, 1) < self.p:
            img = img.transpose(Image.ROTATE_180)
        return img


# noinspection PyPep8Naming
def update(img, oriImg_mean, oriImg_std):
    # img = img.astype(np.uint8)
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hlsImg_mean, hlsImg_std = hlsImg.mean(axis=(0, 1)), hlsImg.std(axis=(0, 1))
    hlsImg[..., 1] = oriImg_std[1] * (hlsImg[..., 1] - hlsImg_mean[1]) / hlsImg_std[1] + oriImg_mean[1]
    hlsImg[..., 1][hlsImg[..., 1] > 255] = 255
    hlsImg[..., 1][hlsImg[..., 1] < 0] = 0
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    # lsImg = lsImg.astype(np.uint8)
    return lsImg


class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, cfg=None, mode='train', transform=None, ext='jpg'):
        super().__init__()
        if mode == 'train':
            self.dirs = cfg.train_dir
        elif mode == 'val':
            self.dirs = cfg.val_dir
            # self.scale = reduce(lambda x,y: x*y,cfg.scaling_rates)
            self.scale = 1024
        elif mode == 'test':
            self.files = []
            for root, dirs, files in os.walk(root_path):
                for dir in dirs:
                    if dir == 'JPEGImages':
                        self.files.extend(glob.glob(os.path.join(root, dir, f'*.{ext}')))
            self.scale = 1024
        else:
            raise NameError('mode must be train or val!')
        if mode == 'test':
            pass
        elif isinstance(self.dirs, list):
            self.files = []
            for dir in self.dirs:
                self.files.extend(glob.glob(os.path.join(root_path, dir, f'*.{ext}')))
        elif isinstance(self.dirs, str):
            self.files = glob.glob(os.path.join(root_path, self.dirs, f'*.{ext}'))
            # pprint(os.path.join(root_path,self.dirs))
        else:
            raise NameError("dirs must be list or str!")
        if mode == 'train':
            self.files *= 1
        # self.files = self.files[:10]
        pprint(f'{mode} dataset length: {len(self.files)}')
        self.mode = mode
        self.transform = transform
        self.threshold = 15

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(name)
        if self.transform:
            img = self.transform(img)
        if self.mode in ['val', 'test']:
            _, h, w = img.shape
            H = int(np.ceil(h / self.scale)) * self.scale
            W = int(np.ceil(w / self.scale)) * self.scale
            img = my_resize(img, H, W)
        return img, name

    def get_shape(self, level):
        return self.__getitem__(0)[level].shape


if __name__ == '__main__':
    root_path = 'D:/dataset/FMD/Ori'
    out_path = 'D:/dataset/FMD/Ori/enhancementlight'
    os.makedirs(out_path, exist_ok=True)
    # datasets = AnomalyDataset(root_path)
    # dataloader = torch.utils.data.DataLoader(datasets,batch_size=1,num_workers=8)
    # with open(os.path.join(out_path,'cutting_points.txt'),'w') as f:
    #     for i, (img, name, diff_special) in tqdm.tqdm(enumerate(dataloader)):
    #         cv2.imwrite(os.path.join(out_path,name[0]),img[0].permute((1,2,0)).numpy().astype('uint8'))
    #         line = f'{name[0]} {" ".join(map(lambda x:str(x),diff_special[0].tolist()))}\n'
    #         f.write(line)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(1024),
        RandomRotation(),
        HistogramEqualization(),
        RandomColorJitter(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = torchvision.datasets.DatasetFolder(root_path, transform=transforms)
    # pprint(dataset[0][0])
    # pprint(dataset[0][0].shape)
    cv2.imshow('img', (255 * dataset[0][0]).permute((1, 2, 0)).numpy().astype('uint8'))
    pprint(dataset[0][1])
    cv2.waitKey()
