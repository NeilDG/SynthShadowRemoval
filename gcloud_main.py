#BASE SCRIPT FOR RUNNING IN GLOUD
import os
def main():
    os.system("python \"render_maps_main.py\" --server_config=3 --num_workers=4 --batch_size=256 "
              "--img_to_load=-1 --load_previous=1 --test_mode=0 --num_blocks=6 "
              "--net_config=1 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
              "--version_name=\"rgb2smoothness_v1.05\" --iteration=1 --map_choice=\"smoothness\"")

    os.system("python \"render_maps_main.py\" --server_config=3 --num_workers=4 --batch_size=256 "
              "--img_to_load=-1 --load_previous=1 --test_mode=0 --num_blocks=6 "
              "--net_config=1 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
              "--version_name=\"rgb2smoothness_v1.05\" --iteration=2 --map_choice=\"smoothness\"")

    os.system("python \"render_maps_main.py\" --server_config=3 --num_workers=4 --batch_size=512 "
              "--img_to_load=-1 --load_previous=1 --test_mode=0 --num_blocks=0 "
              "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
              "--version_name=\"rgb2smoothness_v1.06\" --iteration=1 --map_choice=\"smoothness\"")

    os.system("python \"render_maps_main.py\" --server_config=3 --num_workers=4 --batch_size=512 "
              "--img_to_load=-1 --load_previous=0 --test_mode=0 --num_blocks=0 "
              "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
              "--version_name=\"rgb2smoothness_v1.06\" --iteration=2 --map_choice=\"smoothness\"")

if __name__ == "__main__":
    main()