# server = "3090_242"
# server = "PC"
# server = 'auto_dl'
# server = 'zwd'
# server = 'hao_noise'
# server = 'hao_public'

server = 'zwd_public'



def get_path_dict_ImageDehazing():

    if server == "PC":
        path_dict = {
            "OHAZE":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/OHAZE/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/OHAZE/val/"
                },

            "ITS":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/ITS/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/ITS/val/"
                },

            "OTS":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/OTS/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/OTS/val/"
                },

            "RTTS": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/RTTS/hazy/",

            "4KDehazing":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/4KDehazing/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/4KDehazing/val/"
                },

            "NYU_Seg_Haze":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/NYU_Seg_Haze/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/NYU_Seg_Haze/val/"
                },

            "Places365": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/Places365/",

            # Nighttime Dehazing
            "NHR":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/3R/NHR/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/3R/NHR/val/"
                },

            "NHM":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/3R/NHM/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/3R/NHM/val/"
                },

            "UNREAL_NH":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH/val/"
                },

            "UNREAL_NH_NoSky_Dark":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH_NoSky_Dark/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH_NoSky_Dark/val/"
                },
            "RWNHC_MM23_PseudoLabel":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/RWNHC_MM23_PseudoLabel/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/RWNHC_MM23_PseudoLabel/val/"
                },

            "RWNHC_MM23_PseudoLabel_mini":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/RWNHC_MM23_PseudoLabel_mini/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/RWNHC_MM23_PseudoLabel_mini/val/"
                },

            # Remote Sensing Dehazing
            "SateHaze1k_moderate":
                {
                    "train": "F:/CXF_Code/dataset/processed_dataset/rs_dehazing_dataset/Haze1k/Haze1k_moderate/train/",
                    "val": "F:/CXF_Code/dataset/processed_dataset/rs_dehazing_dataset/Haze1k/Haze1k_moderate/test/"
                }
        }

    elif server == "zwd":
        path_dict = {

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_thick/train/", 
                    "val":   "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_thin/train/",
                    "val":   "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_moderate/train/",
                    "val":   "/home/u2021171171/RS_dataset/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "/home/u2021171171/RS_dataset/HRSD/LHID/train/",
                    "val":   "/home/u2021171171/RS_dataset/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "/home/u2021171171/RS_dataset/HRSD/DHID/train/",
                    "val":   "/home/u2021171171/RS_dataset/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "/home/u2021171171/RS_dataset/RICE/RICE1/train",
                    "val":   "/home/u2021171171/RS_dataset/RICE/RICE1/val"
                },
            "RICE2":
                {
                    "train": "/home/u2021171171/RS_dataset/RICE/RICE2/train",
                    "val":   "/home/u2021171171/RS_dataset/RICE/RICE2/val"
                },
            "RSID":
                {
                    "train": "/home/u2021171171/RS_dataset/RSID/train",
                    "val":   "/home/u2021171171/RS_dataset/RSID/val"
                 },
        }
    elif server == "hao_noise":
        path_dict = {

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_thick/train/", 
                    "val":   "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_thin/train/",
                    "val":   "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_moderate/train/",
                    "val":   "/home/u2021010128/RS_dataset/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "/home/u2021010128/RS_dataset/HRSD/LHID/train/",
                    "val":   "/home/u2021010128/RS_dataset/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "/home/u2021010128/RS_dataset/HRSD/DHID/train/",
                    "val":   "/home/u2021010128/RS_dataset/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "/home/u2021010128/RS_dataset/RICE/RICE1/train",
                    "val":   "/home/u2021010128/RS_dataset/RICE/RICE1/val"
                },
            "RICE2":
                {
                    "train": "/home/u2021010128/RS_dataset/RICE/RICE2/train",
                    "val":   "/home/u2021010128/RS_dataset/RICE/RICE2/val"
                },
            "RSID":
                {
                    "train": "/home/u2021010128/RS_dataset/RSID/train",
                    "val":   "/home/u2021010128/RS_dataset/RSID/val"
                 },
        }

    elif server == "hao_public":
        path_dict = {

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_thick/train/", 
                    "val":   "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_thin/train/",
                    "val":   "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_moderate/train/",
                    "val":   "/users/u2021010128/RS_dataset/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "/users/u2021010128/RS_dataset/HRSD/LHID/train/",
                    "val":   "/users/u2021010128/RS_dataset/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "/users/u2021010128/RS_dataset/HRSD/DHID/train/",
                    "val":   "/users/u2021010128/RS_dataset/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "/users/u2021010128/RS_dataset/RICE/RICE1/train",
                    "val":   "/users/u2021010128/RS_dataset/RICE/RICE1/val"
                },
            "RICE2":
                {
                    "train": "/users/u2021010128/RS_dataset/RICE/RICE2/train",
                    "val":   "/users/u2021010128/RS_dataset/RICE/RICE2/val"
                },
            "RSID":
                {
                    "train": "/users/u2021010128/RS_dataset/RSID/train",
                    "val":   "/users/u2021010128/RS_dataset/RSID/val"
                 },
        }
    elif server == "zwd_public":
        path_dict = {

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_thick/train/", 
                    "val":   "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_thin/train/",
                    "val":   "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_moderate/train/",
                    "val":   "/users/u2021171171/RS_dataset/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "/users/u2021171171/RS_dataset/HRSD/LHID/train/",
                    "val":   "/users/u2021171171/RS_dataset/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "/users/u2021171171/RS_dataset/HRSD/DHID/train/",
                    "val":   "/users/u2021171171/RS_dataset/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "/users/u2021171171/RS_dataset/RICE/RICE1/train",
                    "val":   "/users/u2021171171/RS_dataset/RICE/RICE1/val"
                },
            "RICE2":
                {
                    "train": "/users/u2021171171/RS_dataset/RICE/RICE2/train",
                    "val":   "/users/u2021171171/RS_dataset/RICE/RICE2/val"
                },
            "RSID":
                {
                    "train": "/users/u2021171171/RS_dataset/RSID/train",
                    "val":   "/users/u2021171171/RS_dataset/RSID/val"
                 },
        }
    elif server == "auto_dl":
        path_dict = {

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "/root/autodl-tmp/Haze1k/SateHaze1k_thick/train/",
                    "val": "/root/autodl-tmp/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "/root/autodl-tmp/Haze1k/SateHaze1k_thin/train/",
                    "val": "/root/autodl-tmp/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "/root/autodl-tmp/Haze1k/SateHaze1k_moderate/train/",
                    "val": "/root/autodl-tmp/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "/root/autodl-tmp/HRSD/LHID/train/",
                    "val": "/root/autodl-tmp/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "/root/autodl-tmp/HRSD/DHID/train/",
                    "val": "/root/autodl-tmp/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "/root/autodl-tmp/RICE/RICE1/train",
                    "val":   "/root/autodl-tmp/RICE/RICE1/val"
                },
            "RICE2":
                {
                    "train": "/root/autodl-tmp/RICE/RICE2/train",
                    "val":   "/root/autodl-tmp/RICE/RICE2/val"
                },
            "RSID":
                {
                    "train": "/root/autodl-tmp/RSID/train",
                    "val":   "/root/autodl-tmp/RSID/val"
                }
        }

    elif server == "3090_242":
        path_dict = {
            "OHAZE":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/OHAZE/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/OHAZE/val/"
                },

            "OHAZE_val10":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/OHAZE_val10/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/OHAZE_val10/val/"
                },

            "DenseHaze":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/DenseHaze/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/DenseHaze/val/"
                },

            "NHHAZE":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/NHHAZE/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/NHHAZE/val/"
                },

            "ITS":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/RESIDE/ITS/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/RESIDE/ITS/val/"
                },

            "OTS":
                {
                    "train": "/home/hdd/cong_processed_dataset/dehazing_dataset/RESIDE/OTS/train/",
                    "val": "/home/hdd/cong_processed_dataset/dehazing_dataset/RESIDE/OTS/val/"
                },

            "RTTS": "dataset/cong_processed_dataset/dehazing_dataset/RESIDE/RTTS/hazy/",

            "4KDehazing":
                {
                    "train": "/dataset/cong_processed_dataset/dehazing_dataset/4KDehazing/train/",
                    "val": "/dataset/cong_processed_dataset/dehazing_dataset/4KDehazing/val/"
                },

            "NYU_Seg_Haze":
                {
                    "train": "/dataset/cong_processed_dataset/dehazing_dataset/NYU_Seg_Haze/train/",
                    "val": "/dataset/cong_processed_dataset/dehazing_dataset/NYU_Seg_Haze/val/"
                },

            "Places365": "/dataset/cong_processed_dataset/dehazing_dataset/Places365/",

            "NHR":
                {
                    "train": "../../night_dehazing_dataset/3R/NHR/train/",
                    "val": "../../night_dehazing_dataset/3R/NHR/val/"
                },

            "NHM":
                {
                    "train": "../../night_dehazing_dataset/3R/NHM/train/",
                    "val": "../../night_dehazing_dataset/3R/NHM/val/"
                },

            "NHCL":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCL/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCL/val/"
                },

            "NHCM":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCM/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCM/val/"
                },

            "NHCD":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCD/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCD/val/"
                },

            "UNREAL_NH":
                {
                    "train": "../../night_dehazing_dataset/UNREAL_NH/train/",
                    "val": "../../night_dehazing_dataset/UNREAL_NH/val/"
                },

            "UNREAL_NH_NoSky_Dark":
                {
                    "train": "../../night_dehazing_dataset/UNREAL_NH_NoSky_Dark/train/",
                    "val": "../../night_dehazing_dataset/UNREAL_NH_NoSky_Dark/val/"
                },

            "GTA5":
                {
                    "train": "../../night_dehazing_dataset/GTA5/train/",
                    "val": "../../night_dehazing_dataset/GTA5/val/"
                },

            "NightHaze":
                {
                    "train": "../../night_dehazing_dataset/HDP_dataset/NightHaze/train/",
                    "val": "../../night_dehazing_dataset/HDP_dataset/NightHaze/val/"
                },

            "YellowHaze":
                {
                    "train": "../../night_dehazing_dataset/HDP_dataset/YellowHaze/train/",
                    "val": "../../night_dehazing_dataset/HDP_dataset/YellowHaze/val/"
                },

            "RWNHC_MM23_PseudoLabel":
                {
                    "train": "../../night_dehazing_dataset/RWNHC_MM23_PseudoLabel/train/",
                    "val": "../../night_dehazing_dataset/RWNHC_MM23_PseudoLabel/val/"
                },

            # Remote Sensing Dehazing
            "SateHaze1k_thick":
                {
                    "train": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_thick/train/",
                    "val": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_thick/test/"
                },
            "SateHaze1k_thin":
                {
                    "train": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_thin/train/",
                    "val": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_thin/test/"
                },
            "SateHaze1k_moderate":
                {
                    "train": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_moderate/train/",
                    "val": "../../rs_dehazing_dataset/Haze1k/SateHaze1k_moderate/test/"
                },
            "LHID":
                {
                    "train": "../../rs_dehazing_dataset/HRSD/LHID/train/",
                    "val": "../../rs_dehazing_dataset/HRSD/LHID/test/"
                },
            "DHID":
                {
                    "train": "../../rs_dehazing_dataset/HRSD/DHID/train/",
                    "val": "../../rs_dehazing_dataset/HRSD/DHID/test/"
                },
            "RICE1":
                {
                    "train": "../../rs_dehazing_dataset/RICE/RICE1/train/",
                    "val": "../../rs_dehazing_dataset/RICE/RICE1/val/"
                },
            "RICE2":
                {
                    "train": "../../rs_dehazing_dataset/RICE/RICE2/train/",
                    "val": "../../rs_dehazing_dataset/RICE/RICE2/val/"
                },
            "RSID":
                {
                    "train": "../../rs_dehazing_dataset/RSID/train/",
                    "val": "../../rs_dehazing_dataset/RSID/val/"
                }
        }

    else:
        path_dict = None
    return path_dict


def get_path_dict_UWIE():
    if server == "PC":
        path_dict = {
            "UIEB":
                {"train": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/UIEB/train/",
                 "val": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/UIEB/val/"},

            "EUVP_D":
                {"train": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_D/train/",
                 "val": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_D/val/"},

            "EUVP_I": {
                "train": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_I/train/",
                "val": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_I/val/"},

            "EUVP_S": {
                "train": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_S/train/",
                "val": "F:/CXF_Code/dataset/processed_dataset/underwater_dataset/EUVP/EUVP_S/val/"}}

    elif server == "3090":
        path_dict = {
            "UIEB":
                {"train": "../../dataset/processed_dataset/underwater_dataset/UIEB/train/",
                 "val": "../../dataset/processed_dataset/underwater_dataset/UIEB/val/"},

            "EUVP_D":
                {"train": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_D/train/",
                 "val": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_D/val/"},

            "EUVP_I": {
                "train": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_I/train/",
                "val": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_I/val/"},

            "EUVP_S": {
                "train": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_S/train/",
                "val": "../../dataset/processed_dataset/underwater_dataset/EUVP/EUVP_S/val/"}}

    else:
        path_dict = None

    return path_dict


def get_path_dict_VideoDehazing():
    if server == "PC":
        path_dict = {"REVIDE":
                         {"train": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/REVIDE_Video/train/",
                          "val": "F:/CXF_Code/dataset/processed_dataset/dehazing_dataset/REVIDE_Video/val/"}
                     }

    elif server == "School":
        path_dict = {"REVIDE":
                         {"train": "../../dataset/processed_dataset/dehazing_dataset/REVIDE_Video/train/",
                          "val": "../../dataset/processed_dataset/dehazing_dataset/REVIDE_Video/val/"}
                     }

    elif server == "3090":
        path_dict = {"REVIDE":
                         {"train": "/home/dataset/cong_processed_dataset/dehazing_dataset/REVIDE_Video/train/",
                          "val": "/home/dataset/cong_processed_dataset/dehazing_dataset/REVIDE_Video/val/"}
                     }

    else:
        path_dict = None

    return path_dict