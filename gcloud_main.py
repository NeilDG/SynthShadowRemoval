#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def train_albedo():
    #train albedo
    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=50 --light_angle=0 "
              "--version_name=\"rgb2albedo_v7.16\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=60 --light_angle=36 "
              "--version_name=\"rgb2albedo_v7.16\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=70 --light_angle=72 "
              "--version_name=\"rgb2albedo_v7.16\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=80 --light_angle=108 "
              "--version_name=\"rgb2albedo_v7.16\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=90 --light_angle=144 "
              "--version_name=\"rgb2albedo_v7.16\" --iteration=1")

def train_shading():
    #train shading
    os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=50 --light_angle=0 "
              "--version_name=\"rgb2shading_v8.06\" --iteration=1")

    os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=60 --light_angle=36 "
              "--version_name=\"rgb2shading_v8.06\" --iteration=1")

    os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=70 --light_angle=72 "
              "--version_name=\"rgb2shading_v8.06\" --iteration=1")

    os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=80 --light_angle=108 "
              "--version_name=\"rgb2shading_v8.06\" --iteration=1")

    os.system("python \"shading_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=90 --light_angle=144 "
              "--version_name=\"rgb2shading_v8.06\" --iteration=1")

def train_shadow():
    # train shadow
    os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=50 --light_angle=0 "
              "--version_name=\"rgb2shadow_v8.06\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=60 --light_angle=36 "
              "--version_name=\"rgb2shadow_v8.06\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=70 --light_angle=72 "
              "--version_name=\"rgb2shadow_v8.06\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=80 --light_angle=108 "
              "--version_name=\"rgb2shadow_v8.06\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=90 --light_angle=144 "
              "--version_name=\"rgb2shadow_v8.06\" --iteration=1")

def main():
    # train_albedo()
    train_shading()
    train_shadow()


if __name__ == "__main__":
    main()