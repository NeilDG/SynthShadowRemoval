import os
def main():
    # os.system("python \"embedding_main.py\" --num_workers=6 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=64 --num_blocks=3 "
    #           "--likeness_weight=10.0 --use_bce=0 --use_lpips=1 "
    #           "--version_name=\"embedding_v1.00\" --iteration=1")
    # #

    os.system("python \"render_segment_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=256 --num_blocks=6 "
              "--net_config=4 --l1_weight=10.0 --use_bce=0 "
              "--version_name=\"rgb2smoothness_v1.07\" --iteration=1 --map_choice=\"smoothness\"")

    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=128 --num_blocks=9 "
    #           "--net_config=1 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
    #           "--version_name=\"rgb2albedo_v1.05\" --iteration=2 --map_choice=\"albedo\"")
    #
    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --batch_size=128 --num_blocks=3 "
    #           "--net_config=3 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
    #           "--version_name=\"rgb2albedo_v1.06\" --iteration=1 --map_choice=\"albedo\"")
    #
    # os.system("python \"render_maps_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --batch_size=128 --num_blocks=3 "
    #           "--net_config=3 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
    #           "--version_name=\"rgb2albedo_v1.06\" --iteration=2 --map_choice=\"albedo\"")

    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --batch_size=256 --num_blocks=0 "
    #           "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
    #           "--version_name=\"maps2rgb_v1.01\" --iteration=1")
    #
    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=1 --batch_size=256 --num_blocks=0 "
    #           "--net_config=2 --l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
    #           "--version_name=\"maps2rgb_v1.01\" --iteration=2")
    #
    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --test_mode=0 --batch_size=256 --num_blocks=0 "
    #           "--net_config=2 --l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 "
    #           "--version_name=\"maps2rgb_v1.01\" --iteration=3")
    #
    # os.system("python \"relighting_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=256 --num_blocks=0 "
    #           "--net_config=2 --l1_weight=1.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=1 "
    #           "--version_name=\"maps2rgb_v1.01\" --iteration=4")

if __name__ == "__main__":
    main()