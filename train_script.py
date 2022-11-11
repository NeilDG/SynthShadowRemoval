#Script to use for running heavy training.

import os
def train_relighting():
    os.system("python \"iid_train_v3.py\" --server_config=5 --img_to_load=1000 --train_mode=train_shadow_matte "
              "--plot_enabled=0  --shadow_matte_network_version=\"v58.65\" --shadow_removal_version=\"v58.28\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=1")

def train_domain_adaptation():
    os.system("python \"cyclegan_main.py\" --server_config=5 --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v10.07\" --iteration=1")

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
    # train_relighting()
    train_domain_adaptation()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()