#Script to use for testing

import os
def main():
    # os.system("python \"processing/dataset_creator.py\"")

    # os.system("python \"defer_render_test.py\" --shadow_multiplier=1.0 --shading_multiplier=1.0")
    #

    os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
              "--net_config_a=1 --num_blocks_a=6 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=2 --num_blocks_s3=0 "
              "--version_albedo=\"rgb2albedo_v7.19\" --iteration_a=5 "
              "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
              "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=5 "
              "--version_shadow_remap=\"shadow2relight_v1.04\" --iteration_s3=5 "
              "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=1 --num_blocks_a=6 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=2 --num_blocks_s3=0 "
    #           "--version_albedo=\"rgb2albedo_v7.19\" --iteration_a=6 "
    #           "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=5 "
    #           "--version_shadow_remap=\"shadow2relight_v1.04\" --iteration_s3=5 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=1 --num_blocks_a=6 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=2 --num_blocks_s3=0 "
    #           "--version_albedo=\"rgb2albedo_v7.19\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=5 "
    #           "--version_shadow_remap=\"shadow2relight_v1.04\" --iteration_s3=5 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=1 --num_blocks_a=6 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=2 --num_blocks_s3=0 "
    #           "--version_albedo=\"rgb2albedo_v7.19\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=5 "
    #           "--version_shadow_remap=\"shadow2relight_v1.04\" --iteration_s3=5 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    # #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=6 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")


if __name__ == "__main__":
    main()