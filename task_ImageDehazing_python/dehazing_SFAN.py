# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import torch.optim as optim
import torch.nn as nn
import torch

from methods.ImageDehazing.SFAN import SFAN

from methods.ImageDehazing.SFAN import options_SFAN

import os
from utils.ImageDehazing.writer import LossWriter, plot_all_losses, write_metrics, save_config_as_json, save_best_model, save_cur_model
from dataset.dataloader_ImageDehazing import get_train_val_loader
from utils.ImageDehazing.make_dir import make_train_dir
from methods.ImageDehazing.epoch_eval import eval_SISO, eval_SISO_cell


def train():
    # 5：
    iteration = 0
    best_psnr = 0
    best_ssim = 0
    for epoch in range(config.total_epoches):
        network.train()
        for data in train_loader:
            image_haze = data["hazy"].to(device)
            image_clear = data["gt"].to(device)
            # #################################################
            optimizer.zero_grad()
            generated_image = network(image_haze)

            # # ---------------------------------------------------------------------------
            label_fft = torch.fft.fft2(image_clear, dim=(-2,-1))
            label_fft = torch.stack((label_fft.real, label_fft.imag), -1)

            pred_fft = torch.fft.fft2(generated_image, dim=(-2,-1))
            pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
            # # ---------------------------------------------------------------------------
            fft_loss = loss_func(label_fft, pred_fft) * 0.05

            pixel_loss = loss_func(generated_image, image_clear) 

            loss = fft_loss + pixel_loss

            loss.backward()
            optimizer.step()

            loss_writer.add("loss", loss.item(), iteration)

            # 4：自加
            iteration += 1

            if iteration % 100 == 0:
                print("Iter {}, Loss is {}".format(iteration, loss.item()))

            # #################################################
        scheduler.step()  # 每个epoch执行一次该句，即可更新学习率
        cur_lr = optimizer.param_groups[-1]['lr']
        print("current lr is: {}".format(cur_lr))

        network.eval()
        ssim, psnr = eval_SISO_cell(val_loader=val_loader, network=network, device=device, save_dir=res_dir,
                                    if_save_cat=True, save_type="cat_images")
        write_metrics(os.path.join(res_dir, "metrics/metric.txt"), epoch=epoch, ssim=ssim, psnr=psnr)
        print("SFAN: || Epoch:{} || iterations: {}||, ||PSNR {:.4}||, ||SSIM {:.4}||".format(epoch, iteration, psnr, ssim))

        best_ssim, best_psnr = save_best_model(cur_psnr=psnr, cur_ssim=ssim, best_psnr=best_psnr,
                                               best_ssim=best_ssim, save_dir=res_dir, network=network,
                                               model_name="SFAN", dataset_name=config.dataset)
        if epoch > SAVE_START_EPOCH:
            save_cur_model(save_dir=res_dir, network=network, model_name="SFAN", dataset_name=config.dataset, epochs=epoch)
        # 更新loss图像
        plot_all_losses(losses_path=os.path.join(res_dir, "losses"))

    # ################################################################################## #
    # eval
    eval_SISO(val_loader=val_loader, network=network, device=device, save_dir=res_dir,
              if_eval_best=True, if_eval_last=True, network_name="SFAN", dataset_name=config.dataset)


if __name__ == "__main__":
    # 1：参数定义
    config = options_SFAN.Options().parse()
    SAVE_START_EPOCH = config.save_start_epoch

    # 2：数据准备
    device = torch.device(config.device)
    train_loader, val_loader = get_train_val_loader(dataset=config.dataset, img_h=config.img_h, img_w=config.img_w,
                                                    train_batch_size=config.train_batch_size,
                                                    num_workers=config.num_workers,
                                                    if_flip=True, if_crop=False, crop_h=256, crop_w=256)

    # 创建存储结果的文件夹
    res_dir = config.results_dir
    print("res_dir===>", res_dir)
    make_train_dir(res_dir)
    loss_writer = LossWriter(os.path.join(res_dir, "losses"))
    save_config_as_json(save_path=os.path.join(res_dir, "configs", "config.txt"), config=config)

    # 3:网络定义
    network = SFAN.SFAN().to(device)

    # 4:损失函数和优化器定义
    optimizer = optim.Adam(network.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.step_size, gamma=config.step_gamma)
    loss_func = nn.L1Loss()

    train()
