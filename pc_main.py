import os
def main():
    # os.system("python \"transfer_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=512 --num_blocks=0 --likeness_weight=1.0 --use_bce=1 --weather=\"night\"")
    os.system("python \"paired_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=0 --batch_size=128 --num_blocks=4 --likeness_weight=10.0 --use_bce=0 --net_config=3 --version_name=\"synthplaces2night_v1.00\" --weather=\"night\"")

if __name__ == "__main__":
    main()