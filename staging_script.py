#Script to use for testing

import os
def main():
    # os.system("python \"processing/dataset_creator.py\"")

    # os.system("python \"defer_render_test.py\" --shadow_multiplier=1.0 --shading_multiplier=1.0")

    os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
              "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
              "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=5 "
              "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
              "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
              "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
              "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
              "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=6 "
              "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
              "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
              "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
              "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
              "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=7 "
              "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
              "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
              "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
              "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
              "--version_albedo=\"rgb2albedo_v7.20\" --iteration_a=8 "
              "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
              "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
              "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=5 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=2 --num_blocks_s2=0 "
    #           "--version_albedo=\"rgb2albedo_v7.22\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=2 "
    #           "--version_shadow=\"rgb2shadow_v8.06\" --iteration_s2=2 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=001")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=5 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.18\" --iteration_a=2 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=4 --num_blocks_a=6 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.17\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.16\" --iteration_a=5 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.16\" --iteration_a=6 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.16\" --iteration_a=7 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")
    #
    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=2 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 "
    #           "--version_albedo=\"rgb2albedo_v7.16\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.06\" --iteration_s1=1 "
    #           "--version_shadow=\"rgb2shadow_v8.05\" --iteration_s2=1 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")


if __name__ == "__main__":
    main()