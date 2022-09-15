import os

def main():
    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=9")

    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=10")

    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=1")

    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=2")

    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=3")

    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v7.10\" --iteration=4")

if __name__ == "__main__":
    main()