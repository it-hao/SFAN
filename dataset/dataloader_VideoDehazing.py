import os
import random
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as FF
from dataset_path_config import get_path_dict_VideoDehazing


class REVIDE_Train(data.Dataset):
    def __init__(self, hazy_dir, clear_dir, seq_length, tar_img_h, tar_img_w):
        super(REVIDE_Train, self).__init__()
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.seq_length = seq_length

        self.crop_rate = 0.5

        self.tar_img_h = tar_img_h
        self.tar_img_w = tar_img_w

        # 统计REVIDE的所有场景
        self.hazy_scene_list = os.listdir(self.hazy_dir)

    def get_seq(self):
        """根据REVIDE的存放方式，随机选择一个场景，并根据seq_length选择对应帧的起止位置"""
        hazy_scene_name = self.hazy_scene_list[random.randint(0, len(self.hazy_scene_list)) - 1]
        all_frames = os.listdir(os.path.join(self.hazy_dir, hazy_scene_name))

        # 随机选择视频片段的起止位置
        start_position = random.randint(0, len(all_frames) - self.seq_length)
        end_position = start_position + self.seq_length

        # print("s e position ", start_position, end_position)

        shape = np.array(Image.open(os.path.join(self.hazy_dir, hazy_scene_name, all_frames[0]))).shape

        hazy_imgs = np.zeros(shape=(self.seq_length, self.tar_img_h, self.tar_img_w, 3))
        clear_imgs = np.zeros(shape=(self.seq_length, self.tar_img_h, self.tar_img_w, 3))

        # 每次读取一张图片，加入到容器中
        for p, i in enumerate(range(start_position, end_position)):
            h_img = Image.open(os.path.join(self.hazy_dir, hazy_scene_name, all_frames[i]))
            h_img = h_img.resize((self.tar_img_h, self.tar_img_w))
            c_img = Image.open(os.path.join(self.clear_dir, hazy_scene_name, all_frames[i]))
            c_img = c_img.resize((self.tar_img_h, self.tar_img_w))

            hazy_imgs[p] = np.array(h_img).astype(np.float32)
            clear_imgs[p] = np.array(c_img).astype(np.float32)

        hazy_imgs = torch.from_numpy(hazy_imgs).type(torch.FloatTensor) / 255.0
        clear_imgs = torch.from_numpy(clear_imgs).type(torch.FloatTensor) / 255.0

        hazy_imgs = hazy_imgs.permute(0, 3, 1, 2)
        clear_imgs = clear_imgs.permute(0, 3, 1, 2)

        # 控制裁剪比例，防止裁剪的patch太小
        crop_h, crop_w = self.crop_rate * shape[1], self.crop_rate * shape[2]
        if crop_h < self.tar_img_h:
            crop_h = self.tar_img_h
        if crop_w < self.tar_img_w:
            crop_w = self.tar_img_w

        # 进行数据增强
        if random.random() > 0.5:
            hazy_imgs = FF.hflip(hazy_imgs)
            clear_imgs = FF.hflip(clear_imgs)

        if random.random() > 0.5:
            hazy_imgs = FF.vflip(hazy_imgs)
            clear_imgs = FF.vflip(clear_imgs)

        # if random.random() > 0.5:
        #     i, j, h, w = tfs.RandomCrop.get_params(hazy_imgs, output_size=(int(crop_h), int(crop_w)))
        #     hazy_imgs = FF.crop(hazy_imgs, i, j, h, w)
        #     clear_imgs = FF.crop(clear_imgs, i, j, h, w)

        # hazy_imgs = FF.resize(hazy_imgs, size=[self.tar_img_h, self.tar_img_w])
        # clear_imgs = FF.resize(clear_imgs, size=[self.tar_img_h, self.tar_img_w])
        return hazy_imgs, clear_imgs

    def __getitem__(self, i):
        hazy_imgs, clear_imgs = self.get_seq()
        # plt.imshow(clear_imgs[0] / 255.0)
        # plt.show()
        #

        return hazy_imgs, clear_imgs

    def __len__(self):
        return len(self.hazy_scene_list)


class REVIDE_Test(data.Dataset):
    def __init__(self, hazy_dir, clear_dir, tar_img_h, tar_img_w):
        super(REVIDE_Test, self).__init__()
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir

        self.tar_img_h = tar_img_h
        self.tar_img_w = tar_img_w

        # 统计REVIDE的所有场景
        self.hazy_scene_list = os.listdir(self.hazy_dir)

    def get_one_video(self, hazy_scene_name):
        """根据REVIDE的存放方式，根据给定场景，读取整个视频"""
        all_frames = os.listdir(os.path.join(self.hazy_dir, hazy_scene_name))

        # 随机选择视频片段的起止位置

        hazy_imgs = np.zeros(shape=(len(all_frames), self.tar_img_h, self.tar_img_w, 3))
        clear_imgs = np.zeros(shape=(len(all_frames), self.tar_img_h, self.tar_img_w, 3))

        # 每次读取一张图片，加入到容器中
        for p in range(0, len(all_frames)):
            h_img = Image.open(os.path.join(self.hazy_dir, hazy_scene_name, all_frames[p]))
            h_img = h_img.resize((self.tar_img_h, self.tar_img_w))
            c_img = Image.open(os.path.join(self.clear_dir, hazy_scene_name, all_frames[p]))
            c_img = c_img.resize((self.tar_img_h, self.tar_img_w))

            hazy_imgs[p] = np.array(h_img).astype(np.float32)
            clear_imgs[p] = np.array(c_img).astype(np.float32)
        return hazy_imgs, clear_imgs, all_frames

    def __getitem__(self, i):
        hazy_imgs, clear_imgs, all_frame_names = self.get_one_video(self.hazy_scene_list[i])
        # plt.imshow(clear_imgs[0] / 255.0)
        # plt.show()
        #
        hazy_imgs = torch.from_numpy(hazy_imgs).type(torch.FloatTensor) / 255.0
        clear_imgs = torch.from_numpy(clear_imgs).type(torch.FloatTensor) / 255.0

        hazy_imgs = hazy_imgs.permute(0, 3, 1, 2)
        clear_imgs = clear_imgs.permute(0, 3, 1, 2)

        data = {"hazy_imgs": hazy_imgs, "clear_imgs": clear_imgs, "scene_name": self.hazy_scene_list[i],
                "all_frame_names": all_frame_names}

        return data

    def __len__(self):
        return len(self.hazy_scene_list)


def load_train_data(batch_size, hazy_dir, clear_dir, num_workers, seq_len, img_size):
    train_set = REVIDE_Train(hazy_dir=hazy_dir, clear_dir=clear_dir,
                             seq_length=seq_len, tar_img_h=img_size[1], tar_img_w=img_size[2])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader_train


def load_test_data(batch_size, hazy_dir, clear_dir, num_workers, seq_len, img_size):
    test_set = REVIDE_Test(hazy_dir=hazy_dir, clear_dir=clear_dir,
                           tar_img_h=img_size[1], tar_img_w=img_size[2])

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_test


def get_train_val_loader(dataset, img_h, img_w, train_batch_size, seq_len,
                         num_workers, if_flip, if_crop, crop_h=0, crop_w=0):
    supported_dataset = {
        "REVIDE_train": REVIDE_Train,
        "REVIDE_val": REVIDE_Test
    }

    path_dict = get_path_dict_VideoDehazing()

    try:
        data_root_train = path_dict[dataset]["train"]
        data_root_val = path_dict[dataset]["val"]

    except:
        raise ValueError("dataset not support")
    img_size = [img_h, img_w]
    crop_size = [crop_h, crop_w]

    train_set = supported_dataset[dataset + "_train"](hazy_dir=os.path.join(data_root_train, "hazy"),
                                                      clear_dir=os.path.join(data_root_train, "clear"),
                                                      seq_length=seq_len, tar_img_h=img_size[0], tar_img_w=img_size[1])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    test_set = supported_dataset[dataset + "_val"](hazy_dir=os.path.join(data_root_val, "hazy"),
                                                   clear_dir=os.path.join(data_root_val, "clear"),
                                                   tar_img_h=img_size[0], tar_img_w=img_size[1])

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_test


if __name__ == "__main__":
    # ********************************************************************************* #
    # 训练Loader
    # ds = REVIDE_Train(hazy_dir="E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/REVIDE_Video/train/hazy/",
    #                   clear_dir="E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/REVIDE_Video/train/clear/",
    #                   seq_length=5, tar_img_h=256, tar_img_w=256)
    #
    # for h, c in ds:
    #     print(h.size(), c.size())
    #     h = h.permute(0, 2, 3, 1)  # from (5, 3, 256, 256) to (5, 256, 256, 3)
    #     c = c.permute(0, 2, 3, 1)
    #     # print(h.size(), c.size())
    #
    #     fig = plt.figure(figsize=(10, 2))
    #     for i in range(5):
    #         plt.subplot(1, 5, i + 1)
    #         plt.imshow(c[i].numpy())
    #     plt.show()

    # ********************************************************************************* #

    # 测试Loader
    dataloader_train, dataloader_test = get_train_val_loader(dataset="REVIDE", img_h=256, img_w=256,
                                                             train_batch_size=2, seq_len=5,
                         num_workers=0, if_flip=True, if_crop=True, crop_h=0, crop_w=0)
    dataloader_test = list(dataloader_test)
    h, c, name, all_frame_names = dataloader_test[0]["hazy_imgs"], dataloader_test[0]["clear_imgs"], \
                                  dataloader_test[0]["scene_name"], dataloader_test[0]["all_frame_names"]
    print(h.size(), c.size())

    h = h.permute(0, 2, 3, 1)  # from (5, 3, 256, 256) to (5, 256, 256, 3)
    c = c.permute(0, 2, 3, 1)
    print(h.size(), c.size())

    plt.imshow(c[4].numpy())
    plt.show()

    print("scene name: ", name)
    print("all frame names: ", all_frame_names)

