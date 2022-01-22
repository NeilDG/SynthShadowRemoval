#Script to use for running heavy training.

import os
def train_albedo():
    #train albedo
    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2albedo_v7.15\" --iteration=5")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2albedo_v7.15\" --iteration=5")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2albedo_v7.15\" --iteration=5")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=400 --light_angle=108 "
              "--version_name=\"rgb2albedo_v7.15\" --iteration=5")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=500 --light_angle=144 "
              "--version_name=\"rgb2albedo_v7.15\" --iteration=5")

def train_shading():
    #train shading
    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=400 --light_angle=108 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=500 --light_angle=144 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

def train_shadow():
    # train shadow
    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shadow_v8.05\" --iteration=6")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shadow_v8.05\" --iteration=6")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shadow_v8.05\" --iteration=6")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=400 --light_angle=108 "
              "--version_name=\"rgb2shadow_v8.05\" --iteration=6")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=500 --light_angle=144 "
              "--version_name=\"rgb2shadow_v8.05\" --iteration=6")

def main():
    train_albedo()
    train_shading()
    train_shadow()


if __name__ == "__main__":
    main()