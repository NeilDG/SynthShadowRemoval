#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def train_albedo():
    #train albedo
    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=4 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.17\" --iteration=5")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=4 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.17\" --iteration=6")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=4 --num_blocks=6 "
              "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.17\" --iteration=7")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=4 --num_blocks=6 "
              "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.17\" --iteration=8")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=544 --net_config=5 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.18\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=544 --net_config=5 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.18\" --iteration=2")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=544 --net_config=5 --num_blocks=0 "
              "--l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.18\" --iteration=3")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=544 --net_config=5 --num_blocks=0 "
              "--l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.18\" --iteration=4")
#
# def train_shading():
#     #train shading
#     os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=1 --num_blocks=6 "
#               "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=30 "
#               "--version_name=\"rgb2shading_v8.05\" --iteration=2")
#
# def train_shadow():
#     # train shadow
#     os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=320 --net_config=1 --num_blocks=6 "
#               "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=30 "
#               "--version_name=\"rgb2shadow_v8.05\" --iteration=2")

def main():
    train_albedo()
    # train_shading()
    # train_shadow()


if __name__ == "__main__":
    main()