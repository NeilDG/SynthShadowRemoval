#Script to use for testing

import os

def test_shadow():
    #DEFAULT - HIGHEST PERFORMING - ISTD
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=1 --train_mode=all --dataset_target=istd "
    #           "--shadow_matte_version=\"rgb2sm_v58.28_istd\" --shadow_removal_version=\"rgb2ns_v58.28\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1")
    #
    # # DEFAULT - HIGHEST PERFORMING - SRD
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=1 --train_mode=all --dataset_target=srd "
    #           "--shadow_matte_version=\"rgb2sm_v58.28_srd\" --shadow_removal_version=\"rgb2ns_v58.28\" "
    #           "--shadow_matte_iteration=4 --shadow_removal_iteration=1")
    #

    os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
              "--img_vis_enabled=1 --train_mode=train_shadow_matte --dataset_target=all "
              "--shadow_matte_version=\"rgb2sm_v61.02\" --shadow_removal_version=\"rgb2ns_v58.28\" "
              "--shadow_matte_iteration=1 --shadow_removal_iteration=1")

    # os.system("python \"benchmark_shadow.py\" --img_to_load=-1")

    #FOR TESTING
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=1 --train_mode=train_shadow_matte "
    #           "--shadow_matte_version=\"rgb2sm_v61.01\" --shadow_removal_version=\"rgb2ns_v61.00_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1")


def test_img2img():
    # os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_images=1 --network_version=\"synth2istd_v01.00\" "
    #           "--iteration=4")
    #
    os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_images=1 --network_version=\"synth2srd_v01.00\" "
              "--iteration=4")

def main():
    test_shadow()
    # test_img2img()


if __name__ == "__main__":
    main()
