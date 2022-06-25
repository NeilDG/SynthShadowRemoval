#Script to use for testing

import os

def test_relighting():
    os.system("python \"iid_test.py\"  --net_config=4 --num_blocks=4 --version_name=\"iid_networkv7.04\" --albedo_mode=0 --iteration=10 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\" "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=4 --num_blocks=4 --version_name=\"iid_networkv7.04\" --albedo_mode=0 --iteration=11 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\" "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=4 --num_blocks=4 --version_name=\"iid_networkv7.04\" --albedo_mode=0 --iteration=12 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\" "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=4 --num_blocks=4 --version_name=\"iid_networkv7.04\" --albedo_mode=0 --iteration=13 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\" "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=4 --num_blocks=4 --version_name=\"iid_networkv7.04\" --albedo_mode=0 --iteration=14 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\" "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")


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