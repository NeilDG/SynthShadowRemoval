#Script to use for running heavy training.

import os
def main():
    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    #
    #
    # os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2shadowmap_v7.09\" --iteration=1")
    #
    # os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
    #           "--version_name=\"rgb2shadowmap_v7.09\" --iteration=1")
    #
    # os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
    #           "--version_name=\"rgb2shadowmap_v7.09\" --iteration=1")
    #
    # os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=400 --light_angle=144 "
    #           "--version_name=\"rgb2shadowmap_v7.09\" --iteration=1")

    # os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2shading_v8.01\" --iteration=1")
    #
    # os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=300 --light_angle=36 "
    #           "--version_name=\"rgb2shading_v8.01\" --iteration=1")
    #
    # os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=400 --light_angle=72 "
    #           "--version_name=\"rgb2shading_v8.01\" --iteration=1")
    #
    # os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=500 --light_angle=108 "
    #           "--version_name=\"rgb2shading_v8.01\" --iteration=1")
    #
    # os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=600 --light_angle=144 "
    #           "--version_name=\"rgb2shading_v8.01\" --iteration=1")

    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=100 --light_angle=0 "
    #           "--version_name=\"rgb2shadow_v9.02\" --iteration=1")
    #
    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=200 --light_angle=36 "
    #           "--version_name=\"rgb2shadow_v9.02\" --iteration=1")
    #
    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=300 --light_angle=72 "
    #           "--version_name=\"rgb2shadow_v9.02\" --iteration=1")
    #
    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=400 --light_angle=108 "
    #           "--version_name=\"rgb2shadow_v9.02\" --iteration=1")
    #
    # os.system("python \"shadowmap_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=0 --mode=azimuth --min_epochs=500 --light_angle=144 "
    #           "--version_name=\"rgb2shadow_v9.02\" --iteration=1")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shading_v8.01\" --iteration=2")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shading_v8.01\" --iteration=2")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shading_v8.01\" --iteration=2")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=300 --light_angle=108 "
              "--version_name=\"rgb2shading_v8.01\" --iteration=2")

    os.system("python \"shading_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=400 --light_angle=144 "
              "--version_name=\"rgb2shading_v8.01\" --iteration=2")

    os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=100 --light_angle=0 "
              "--version_name=\"rgb2shadow_v8.01\" --iteration=2")

    os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=200 --light_angle=36 "
              "--version_name=\"rgb2shadow_v8.01\" --iteration=2")

    os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=300 --light_angle=72 "
              "--version_name=\"rgb2shadow_v8.01\" --iteration=2")

    os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=300 --light_angle=108 "
              "--version_name=\"rgb2shadow_v8.01\" --iteration=2")

    os.system("python \"shadowmap_main_old.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=512 --net_config=2 --num_blocks=0 "
              "--l1_weight=10.0 --lpip_weight=0.0 --ssim_weight=1.0 --adv_weight=1.0 --use_bce=1 --mode=azimuth --min_epochs=400 --light_angle=144 "
              "--version_name=\"rgb2shadow_v8.01\" --iteration=2")


if __name__ == "__main__":
    main()