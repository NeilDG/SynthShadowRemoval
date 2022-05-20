import os

def main():
    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=640 --net_config=3 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=9")

    os.system("python \"relighting_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=640 --net_config=3 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=10")

if __name__ == "__main__":
    main()