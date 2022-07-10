import os

def main():
    os.system("python \"cyclegan_main.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --load_previous=1 --test_mode=0 --net_config=2 --num_blocks=0 "
              "--patch_size=32 --img_per_iter=60 --batch_size=1 --plot_enabled=0 "
              "--min_epochs=100 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2rgb_v5.00\" --iteration=6")

if __name__ == "__main__":
    main()