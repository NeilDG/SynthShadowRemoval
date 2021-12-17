import os
def main():
    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    # #

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=1 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=1")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=2 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=2")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=3 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=3")
    #
    # os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=1 --batch_size=8 "
    #           "--net_config_a=3 --num_blocks_a=3 --net_config_s=3 --num_blocks_s=3 "
    #           "--version_name_a=\"rgb2albedo_v2.05\" --iteration_a=3 --version_name_s=\"rgb2shading_v2.05\" --iteration_s=3")

    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.04\" --iteration=1 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.04\" --iteration=1 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.04\" --iteration=2 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=0.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.04\" --iteration=2 --map_choice=\"shading\"")

    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.04\" --iteration=3 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=1 --batch_size=256 --net_config=1 --num_blocks=6 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.04\" --iteration=3 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.02\" --iteration=4 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=1.0 --lpip_weight=10.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.02\" --iteration=4 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.02\" --iteration=5 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.02\" --iteration=5 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.02\" --iteration=6 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=0.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.02\" --iteration=6 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.02\" --iteration=7 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=0 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.02\" --iteration=7 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2albedo_v2.02\" --iteration=8 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --net_config=3 --num_blocks=3 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --use_bce=1 --min_epochs=200 "
    #           "--version_name=\"rgb2shading_v2.02\" --iteration=8 --map_choice=\"shading\"")

if __name__ == "__main__":
    main()