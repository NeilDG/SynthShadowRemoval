import os

def main():
    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=192 --min_epochs=40 "
              "--net_config=4 --num_blocks=4 "
              "--plot_enabled=0 --debug_mode=0 --version_name=\"iid_networkv8.04\" --iteration=5 "
              "--unlit_checkpt_file=\"synth2unlit_v1.00_1.pt\" --da_enabled=1 --da_version_name=\"embedding_v5.00_5\" --albedo_mode=2")

if __name__ == "__main__":
    main()