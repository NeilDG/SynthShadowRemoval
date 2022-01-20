#BASE SCRIPT FOR RUNNING IN GLOUD
import os
def main():
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --batch_size=256 --net_config=5 --num_blocks=8 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v3.01\" --iteration=99 --map_choice=\"shading\"")

    # os.system("python \"shading_main_albedo.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2shading_v7.05\" --iteration=1")
    #
    # os.system("python \"shadowmap_main_old.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2shadowmap_v7.05\" --iteration=1")

    # os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=32 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2albedo_v7.12\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=40 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2albedo_v7.12\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=40 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2albedo_v7.12\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=40 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=400 --light_angle=108 "
              "--version_name=\"rgb2albedo_v7.12\" --iteration=1")

    os.system("python \"albedo_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=40 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --use_mask=0 --mode=azimuth --min_epochs=500 --light_angle=144 "
              "--version_name=\"rgb2albedo_v7.12\" --iteration=1")


if __name__ == "__main__":
    main()