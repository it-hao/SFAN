import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as tt
import random
from torchvision.transforms import functional as FF
import torchvision.transforms as tfs
import torch
from dataset_path_config import get_path_dict_UWIE


def pair_augmentation(img, target, new_size):
    if random.random() > 0.5:
        img = FF.hflip(img)
        target = FF.hflip(target)
    if random.random() > 0.5:
        img = FF.vflip(img)
        target = FF.vflip(target)

    if random.random() > 0.5:
        i, j, h, w = tfs.RandomCrop.get_params(img, output_size=(new_size[0], new_size[1]))
        img = FF.crop(img, i, j, h, w)
        target = FF.crop(target, i, j, h, w)

    return img, target


class PairDataset(data.Dataset):
    def __init__(self, images_path, labels_path, img_size, if_train, trans_hazy=None, trans_gt=None,
                 if_identity_name=False):
        super().__init__()
        self.haze_imgs_dir = os.listdir(images_path)
        self.haze_imgs = [os.path.join(images_path, img) for img in self.haze_imgs_dir]
        # print(self.haze_imgs)

        self.clear_dir = labels_path

        self.img_size = img_size
        self.if_train = if_train

        self.if_identity_name = if_identity_name

        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                          tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                        tt.ToTensor()])

        self.split = "/"
        # if platform.system() == "Windows":
        #     self.split = "\\"

    def __getitem__(self, index):
        data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')
        img = self.haze_imgs[index]
        clear_name = None
        if self.if_identity_name:
            clear_name = img.split(self.split)[-1]

        data_gt = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        # 不进行数据增强，只用在测试集
        if self.if_train:
            if data_hazy.width >= self.img_size[0] and data_hazy.height >= self.img_size[1]:
                data_hazy, data_gt = pair_augmentation(data_hazy, data_gt, self.img_size)

        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)
        tar_data = {"blur": data_hazy, "gt": data_gt,
                    "name": img.split(self.split)[-1],
                    "blur_path": self.haze_imgs[index]}

        return tar_data

    def __len__(self):
        return len(self.haze_imgs)


class PairDatasetCenterCrop(data.Dataset):
    def __init__(self, images_path, labels_path, center_crop_size, if_train, trans_hazy=None, trans_gt=None,
                 if_identity_name=False):
        super().__init__()
        self.haze_imgs_dir = os.listdir(images_path)
        self.haze_imgs = [os.path.join(images_path, img) for img in self.haze_imgs_dir]
        # print(self.haze_imgs)

        self.clear_dir = labels_path

        self.center_crop_size = center_crop_size
        self.if_train = if_train

        self.if_identity_name = if_identity_name

        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.CenterCrop((self.center_crop_size, self.center_crop_size)),
                                        tt.ToTensor()])

        self.split = "/"

    def __getitem__(self, index):
        data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')
        img = self.haze_imgs[index]
        clear_name = None
        if self.if_identity_name:
            clear_name = img.split(self.split)[-1]

        data_gt = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)
        tar_data = {"blur": data_hazy, "gt": data_gt,
                    "name": img.split(self.split)[-1],
                    "blur_path": self.haze_imgs[index]}

        return tar_data

    def __len__(self):
        return len(self.haze_imgs)


def get_train_val_loader(config):
    path_dict = get_path_dict_UWIE()
    try:
        data_root_train = path_dict[config.dataset_name]["train"]
        data_root_val = path_dict[config.dataset_name]["val"]

    except:
        raise ValueError("dataset not support")

    train_dataset = PairDataset(images_path=os.path.join(data_root_train, "images/"),
                                labels_path=os.path.join(data_root_train, "labels/"),
                                img_size=[256, 256], if_train=True,
                                trans_hazy=None, trans_gt=None, if_identity_name=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    print("Train Dataset Reading Completed.")

    test_dataset = PairDataset(images_path=os.path.join(data_root_val, "images/"),
                               labels_path=os.path.join(data_root_val, "labels/"),
                               img_size=[256, 256], if_train=False,
                               trans_hazy=None, trans_gt=None, if_identity_name=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train_batch_size,
                                                  shuffle=False, num_workers=config.num_workers)

    print("Val Dataset Reading Completed.")

    return train_dataloader, test_dataloader


def get_split_val_loader(config):
    path_dict = get_path_dict_UWIE()
    try:
        data_root_val = path_dict[config.dataset_name]["val"]
    except:
        raise ValueError("dataset not support")

    test_dataset = PairDataset(images_path=os.path.join(config.results_dir, "images/"),
                               labels_path=os.path.join(data_root_val, "labels/"),
                               img_size=[256, 256], if_train=False,
                               trans_hazy=None, trans_gt=None, if_identity_name=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.val_batch_size,
                                                  shuffle=False, num_workers=1)

    return test_dataloader