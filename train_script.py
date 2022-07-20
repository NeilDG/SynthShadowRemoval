#Script to use for running heavy training.

import os
def train_relighting():
    os.system("python \"iid_train_v2.py\" --server_config=5 --img_to_load=3000 --debug_run=1 "
              "--plot_enabled=1 --version=\"v9.07\" --iteration=15")

    # os.system("python \"iid_train_v2.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=0 --version=\"v9.07\" --iteration=16")
    #
    # os.system("python \"iid_train_v2.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=0 --version=\"v9.01\" --iteration=5")
    #
    # os.system("python \"iid_train_v2.py\" --server_config=6 --img_to_load=-1 "
    #           "--plot_enabled=0 --version=\"v9.01\" --iteration=6")


def train_domain_adaptation():
    os.system("python \"cyclegan_main.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=2 --num_blocks=0 "
              "--patch_size=32 --img_per_iter=32 --batch_size=1 --plot_enabled=1 " 
              "--min_epochs=30 --g_lr=0.00002 --d_lr=0.00002 --version_name=\"albedo2colored_v1.01\" --iteration=1")

def train_embedding():
    os.system("python \"embedding_main.py\" --server_config=5 --img_to_load=-1 --load_previous=1 --test_mode=0 --net_config=4 --num_blocks=4 "
              "--plot_enabled=0 --patch_size=64 --batch_size=16 --likeness_weight=10.0 --embedding_dist_weight=1.0 "
              "--min_epochs=5 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"embedding_v6.00\" --iteration=1")

    os.system("python \"embedding_main.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=4 --num_blocks=4 "
              "--plot_enabled=0 --patch_size=64 --batch_size=16 --likeness_weight=1.0 --embedding_dist_weight=1.0 "
              "--min_epochs=5 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"embedding_v6.00\" --iteration=2")

def train_unlit():
    # os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=1 --test_mode=0 --net_config=5 --num_blocks=7 "
    #           "--plot_enabled=0 --patch_size=64 --batch_size=128 "
    #           "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=1")
    #
    # os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=5 --num_blocks=7 "
    #           "--plot_enabled=0 --patch_size=64 --batch_size=128 "
    #           "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=2")
    #
    # os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=5 --num_blocks=7 "
    #           "--plot_enabled=0 --patch_size=64 --batch_size=128 "
    #           "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=3")

    os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=1 --test_mode=0 --net_config=5 --num_blocks=7 "
             "--plot_enabled=1 --patch_size=64 --batch_size=128 "
             "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=4")

    os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=5 --num_blocks=7 "
             "--plot_enabled=0 --patch_size=64 --batch_size=128 "
             "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=5")

    os.system("python \"unlit_train.py\" --server_config=5 --img_to_load=-1 --load_previous=0 --test_mode=0 --net_config=5 --num_blocks=7 "
             "--plot_enabled=0 --patch_size=64 --batch_size=128 "
             "--min_epochs=15 --g_lr=0.0002 --d_lr=0.0002 --version_name=\"synth2unlit_v1.04\" --iteration=6")


def main():
    # train_unlit()
    train_relighting()
    # train_domain_adaptation()
    # train_embedding()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()