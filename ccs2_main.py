import os

def main():
    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=64 --net_config=2 --num_blocks=0 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.13\" --iteration=9")

    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=64 --net_config=2 --num_blocks=0 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.13\" --iteration=10")

    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=64 --net_config=2 --num_blocks=0 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.13\" --iteration=11")

    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=64 --net_config=2 --num_blocks=0 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.13\" --iteration=12")

if __name__ == "__main__":
    main()