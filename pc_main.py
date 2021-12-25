import os
def main():
    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    # #
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --patch_size=256 --batch_size=32 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=120 "
    #           "--version_name=\"rgb2albedo_v4.00\" --iteration=1 --map_choice=\"albedo\"")

    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --patch_size=32 --batch_size=2048 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=0 --min_epochs=120 "
    #           "--version_name=\"rgb2shading_v4.00\" --iteration=1 --map_choice=\"shading\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --patch_size=32 --batch_size=2048 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --min_epochs=120 "
    #           "--version_name=\"rgb2albedo_v4.00\" --iteration=2 --map_choice=\"albedo\"")
    #
    # os.system("python \"iid_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --use_mask=0 --patch_size=32 --batch_size=2048 --net_config=2 --num_blocks=0 "
    #           "--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --adv_weight=1.0 --use_bce=1 --min_epochs=120 "
    #           "--version_name=\"rgb2shading_v4.00\" --iteration=2 --map_choice=\"shading\"")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=1 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=1")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=2 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=2")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=3 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=3")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=4 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=4")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=5 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=5")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=6 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=6")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=7 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=7")

    os.system("python \"iid_render_main.py\" --num_workers=12 --img_to_load=-1 --use_mask=0 --patch_size=64 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s=1 --num_blocks_s=6 "
              "--version_name_a=\"rgb2albedo_v2.04\" --iteration_a=8 --version_name_s=\"rgb2shading_v2.04\" --iteration_s=8")


if __name__ == "__main__":
    main()