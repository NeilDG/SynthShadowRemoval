#BASE SCRIPT FOR RUNNING IN GLOUD
import os
def main():
    # os.system("python \"render_segment_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=1024 "
    #           "--net_config=2 --l1_weight=10.0 --use_bce=0 --min_epochs=120 "
    #           "--version_name=\"rgb2smoothness_v1.08\" --iteration=1 --map_choice=\"smoothness\"")

    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.01\" --iteration=1 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.01\" --iteration=1 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2albedo_v2.01\" --iteration=2 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2shading_v2.01\" --iteration=2 --map_choice=\"shading\"")

    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.01\" --iteration=3 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.01\" --iteration=3 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2albedo_v2.01\" --iteration=4 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2shading_v2.01\" --iteration=4 --map_choice=\"shading\"")

    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.01\" --iteration=5 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.01\" --iteration=5 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2albedo_v2.01\" --iteration=6 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2shading_v2.01\" --iteration=6 --map_choice=\"shading\"")

    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.01\" --iteration=7 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.01\" --iteration=7 --map_choice=\"shading\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2albedo_v2.01\" --iteration=8 --map_choice=\"albedo\"")

    os.system("python \"iid_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --net_config=1 --num_blocks=6 "
              "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
              "--version_name=\"rgb2shading_v2.01\" --iteration=8 --map_choice=\"shading\"")

if __name__ == "__main__":
    main()