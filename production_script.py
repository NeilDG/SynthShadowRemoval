#Script to use for running heavy training.

import os
def train_albedo():
    #train albedo
    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #           "--version_name=\"rgb2albedo_v7.15\" --iteration=7")
    #
    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #          "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #          "--version_name=\"rgb2albedo_v7.15\" --iteration=8")

    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #           "--version_name=\"rgb2albedo_v7.16\" --iteration=7")
    #
    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #           "--version_name=\"rgb2albedo_v7.16\" --iteration=8")
    #

    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=128 --net_config=5 --num_blocks=0 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #           "--version_name=\"rgb2albedo_v7.22\" --iteration=7")
    #
    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=128 --net_config=5 --num_blocks=0 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
    #           "--version_name=\"rgb2albedo_v7.22\" --iteration=8")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=128 --net_config=5 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.22\" --iteration=5")

    os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=128 --net_config=5 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --use_mask=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2albedo_v7.22\" --iteration=6")



def train_shading():
    #train shading
    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=5")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=50 "
              "--version_name=\"rgb2shading_v8.05\" --iteration=6")

def train_shadow():
    # train shadow
    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=64 --net_config=1 --num_blocks=6 "
    #           "--mode=azimuth --min_epochs=50 --version_name=\"rgb2shadow_v8.07\" --iteration=5")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=64 --net_config=1 --num_blocks=6 "
              "--mode=azimuth --min_epochs=50 --version_name=\"rgb2shadow_v8.07\" --iteration=6")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=128 --batch_size=64 --net_config=1 --num_blocks=6 "
              "--mode=azimuth --min_epochs=50 --version_name=\"rgb2shadow_v8.07\" --iteration=7")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=128 --batch_size=64 --net_config=1 --num_blocks=6 "
              "--mode=azimuth --min_epochs=50 --version_name=\"rgb2shadow_v8.07\" --iteration=8")

def train_shadow_relight():
    # os.system("python \"shadow_relight_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=1 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
    #           "--mode=azimuth --min_epochs=50 --version_name=\"shadow2relight_v1.07\" --iteration=5")
    #
    # os.system("python \"shadow_relight_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=1 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
    #           "--mode=azimuth --min_epochs=50 --version_name=\"shadow2relight_v1.07\" --iteration=6")

    os.system("python \"shadow_relight_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=0 --test_mode=1 --patch_size=256 --batch_size=16 --net_config=4 --num_blocks=6 "
              "--mode=azimuth --min_epochs=50 --version_name=\"shadow2relight_v1.09\" --iteration=7")

    os.system("python \"shadow_relight_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=0 --test_mode=1 --patch_size=256 --batch_size=16 --net_config=4 --num_blocks=6 "
              "--mode=azimuth --min_epochs=50 --version_name=\"shadow2relight_v1.09\" --iteration=8")

def train_relighting():
    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=48 --net_config=2 --num_blocks=0 "
    #           "--mode=azimuth --min_epochs=50 --version_name=\"maps2rgb_rgb2maps_v2.00\" --iteration=5")
    #
    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=128 --batch_size=48 --net_config=2 --num_blocks=0 "
    #           "--mode=azimuth --min_epochs=50 --version_name=\"maps2rgb_rgb2maps_v2.00\" --iteration=6")

    os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=1 --patch_size=128 --batch_size=48 --net_config=2 --num_blocks=0 "
              "--mode=azimuth --min_epochs=50 --version_name=\"maps2rgb_rgb2maps_v2.00\" --iteration=7")

    os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=1 --patch_size=128 --batch_size=48 --net_config=2 --num_blocks=0 "
              "--mode=azimuth --min_epochs=50 --version_name=\"maps2rgb_rgb2maps_v2.00\" --iteration=8")

def main():
    # train_albedo()
    # train_shading()
    # train_shadow()
    train_shadow_relight()
    # train_relighting()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()