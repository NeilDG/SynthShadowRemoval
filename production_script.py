#Script to use for running heavy training.

import os
def main():
    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    #
    #
    os.system("python \"shading_main_albedo.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shading_v7.07\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shadowmap_v7.07\" --iteration=1")

    os.system("python \"shading_main_albedo.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shading_v7.07\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shadowmap_v7.07\" --iteration=1")

    os.system("python \"shading_main_albedo.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shading_v7.07\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shadowmap_v7.07\" --iteration=1")

    os.system("python \"shading_main_albedo.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=400 --light_angle=144 "
              "--version_name=\"rgb2shading_v7.07\" --iteration=1")

    os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=256 --batch_size=16 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=400 --light_angle=144 "
              "--version_name=\"rgb2shadowmap_v7.07\" --iteration=1")

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=64 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=2 --num_blocks_s2=0 "
    #           "--version_albedo=\"rgb2albedo_v6.01\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v7.01\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadowmap_v7.01\" --iteration_s2=1 "
    #           "--light_angle=0 --light_color=\"225,247,250\"")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=64 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=2 --num_blocks_s2=0 "
    #           "--version_albedo=\"rgb2albedo_v6.01\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v7.01\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadowmap_v7.01\" --iteration_s2=1 "
    #           "--light_angle=36 --light_color=\"225,247,250\"")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=64 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=2 --num_blocks_s2=0 "
    #           "--version_albedo=\"rgb2albedo_v6.01\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v7.01\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadowmap_v7.01\" --iteration_s2=1 "
    #           "--light_angle=72 --light_color=\"225,247,250\"")


if __name__ == "__main__":
    main()