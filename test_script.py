#Script to use for testing

import os

def test_relighting():
    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v48.05\" --iteration=1")

    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v50.01\" --iteration=1")
    #
    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v50.02\" --iteration=1")

    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v50.03\" --iteration=1")
    #
    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v50.04\" --iteration=1")
    #
    # os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
    #           "--plot_enabled=1 --img_vis_enabled=1 --version=\"v50.03\" --iteration=6")

    os.system("python \"iid_test_v3.py\" --server_config=5 --img_to_load=-1 "
              "--plot_enabled=1 --img_vis_enabled=1 --version=\"v51.01\" --iteration=1")

def main():
    # os.system("python \"processing/dataset_creator.py\"")

    # os.system("python \"defer_render_test.py\" --shadow_multiplier=1.0 --shading_multiplier=1.0")
    #

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=5 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=1 --num_blocks_s3=6 "
    #           "--version_albedo=\"rgb2albedo_v7.22\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
    #           "--version_shadow=\"rgb2shadow_v8.07\" --iteration_s2=5 "
    #           "--version_shadow_remap=\"shadow2relight_v1.07\" --iteration_s3=8 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    test_relighting()



if __name__ == "__main__":
    main()