import os

def main():
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=24 --net_config=1 --num_blocks=6 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=9")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=24 --net_config=1 --num_blocks=6 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=10")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=24 --net_config=1 --num_blocks=6 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=11")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=256 --batch_size=24 --net_config=1 --num_blocks=6 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"maps2rgb_rgb2maps_v4.12\" --iteration=12")

    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 "
    #           "--net_config=2 --num_blocks=0 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=9")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 "
    #           "--net_config=2 --num_blocks=0 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=10")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 "
    #           "--net_config=2 --num_blocks=0 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=11")
    #
    # os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=512 "
    #           "--net_config=2 --num_blocks=0 "
    #           "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=12")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 "
              "--net_config=1 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=9")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 "
              "--net_config=1 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=10")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 "
              "--net_config=1 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=11")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 "
              "--net_config=1 --num_blocks=6 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv5.01\" --iteration=12")

if __name__ == "__main__":
    main()