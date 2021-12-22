#BASE SCRIPT FOR RUNNING IN GLOUD
import os
def main():
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.01\" --iteration=8 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.01\" --iteration=8 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --batch_size=256 --net_config=5 --num_blocks=8 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=120 "
              "--version_name=\"rgb2albedo_v3.01\" --iteration=98 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --batch_size=256 --net_config=5 --num_blocks=8 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=120 "
              "--version_name=\"rgb2shading_v3.01\" --iteration=98 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --batch_size=256 --net_config=5 --num_blocks=8 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2albedo_v3.01\" --iteration=99 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --batch_size=256 --net_config=5 --num_blocks=8 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2shading_v3.01\" --iteration=99 --map_choice=\"shading\"")

if __name__ == "__main__":
    main()