import os


def make_train_dir(res_dir):
    # if os.path.exists(res_dir):
        # raise ValueError("res_dir already exists, avoid overwriting !!!!!!")
    if not os.path.exists(res_dir):
        # print("==========================")
        os.makedirs(res_dir)
        os.makedirs(os.path.join(res_dir, "cat_images"))
        os.makedirs(os.path.join(res_dir, "best_PSNR_images"))
        os.makedirs(os.path.join(res_dir, "best_SSIM_images"))
        os.makedirs(os.path.join(res_dir, "last_images"))
        os.makedirs(os.path.join(res_dir, "models"))
        os.makedirs(os.path.join(res_dir, "metrics"))
        os.makedirs(os.path.join(res_dir, "losses"))
        os.makedirs(os.path.join(res_dir, "configs"))
        os.makedirs(os.path.join(res_dir, "sample_images"))