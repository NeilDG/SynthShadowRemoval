import os
def main():
    # os.system("python \"transfer_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=512 --num_blocks=0 --likeness_weight=1.0 --use_bce=1 --weather=\"night\"")
    # os.system("python \"paired_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --iteration=3 --batch_size=128 --num_blocks=6 "
    #           "--likeness_weight=10.0 --use_bce=1 --use_lpips=1 --use_mask=1 "
    #           "--net_config=1 --version_name=\"synthplaces2cloudy_v1.02\" --weather=\"cloudy\"")

    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    # #

    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=128 --num_blocks=9 "
    #           "--net_config=1 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
    #           "--version_name=\"rgb2albedo_v1.05\" --iteration=1 --map_choice=\"albedo\"")

    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=128 --num_blocks=9 "
    #           "--net_config=1 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
    #           "--version_name=\"rgb2albedo_v1.05\" --iteration=2 --map_choice=\"albedo\"")
    #
    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=128 --num_blocks=1 "
    #           "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
    #           "--version_name=\"rgb2albedo_v1.06\" --iteration=1 --map_choice=\"albedo\"")

    os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=128 --num_blocks=1 "
              "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
              "--version_name=\"rgb2albedo_v1.06\" --iteration=2 --map_choice=\"albedo\"")

if __name__ == "__main__":
    main()