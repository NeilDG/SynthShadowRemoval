import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --train_mode=train_shadow_matte --version=\"v58.05\" --iteration=1")

if __name__ == "__main__":
    main()