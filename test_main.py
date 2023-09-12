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

    # FOR TESTING NEW SM + SR COMBINATION - ISTD
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=all --dataset_target=istd "
    #           "--shadow_matte_version=\"rgb2sm_v61.32_istd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")
    #
    # # FOR TESTING NEW SM + SR COMBINATION - SRD
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=all --dataset_target=srd "
    #           "--shadow_matte_version=\"rgb2sm_v61.32_srd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")
    #
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=all --dataset_target=istd "
    #           "--shadow_matte_version=\"rgb2sm_v61.82_istd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")
    #
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=all --dataset_target=srd "
    #           "--shadow_matte_version=\"rgb2sm_v61.82_srd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")

    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=train_shadow_matte --dataset_target=all "
    #           "--shadow_matte_version=\"rgb2sm_v61.85_istd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")
    #
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=train_shadow_matte --dataset_target=all "
    #           "--shadow_matte_version=\"rgb2sm_v61.85_srd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")

    # os.system("python \"shadow_test_main-2.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=1 --train_mode=train_shadow --dataset_target=istd "
    #           "--shadow_matte_version=\"rgb2sm_v61.13_places\" --shadow_removal_version=\"rgb2ns_v61.14_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1")
    #
    os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
              "--img_vis_enabled=0 --train_mode=train_shadow --dataset_target=srd "
              "--shadow_matte_version=\"rgb2sm_v61.32_srd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
              "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=1")


    # os.system("python \"shadow_test_main-3.py\" --server_config=5 --img_to_load=100000 "
    #           "--img_vis_enabled=0 --train_mode=train_shadow --dataset_target=all "
    #           "--shadow_matte_version=\"rgb2sm_v58.28_istd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1")
    #
    #
    # os.system("python \"benchmark_shadow.py\" --img_to_load=-1")

    #FOR TESTING
    # os.system("python \"shadow_test_main.py\" --server_config=5 --img_to_load=-1 "
    #           "--img_vis_enabled=0 --train_mode=train_shadow "
    #           "--shadow_matte_version=\"rgb2sm_v61.01\" --shadow_removal_version=\"rgb2ns_v61.12_places\" "
    #           "--shadow_matte_iteration=1 --shadow_removal_iteration=1")


def analyze():
    os.system("python \"shadow_analyzer_main.py\" --server_config=5 --img_to_load=-1 "
              "--img_vis_enabled=1 --train_mode=train_shadow_matte --dataset_target=all "
              "--shadow_matte_version=\"rgb2sm_v61.64_istd\" --shadow_removal_version=\"rgb2ns_v61.26_places\" "
              "--shadow_matte_iteration=1 --shadow_removal_iteration=1 --load_best=0")
def test_img2img():
    os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_images=0 --network_version=\"synth2srd_v01.00\" "
              "--iteration=5")

    # os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_images=1 --network_version=\"synth2srd_v01.00\" "
    #           "--iteration=4")

def main():
    # analyze()
    test_shadow()
    # test_img2img()


if __name__ == "__main__":
    main()
