#BASE SCRIPT FOR RUNNING IN GLOUD
import os
def main():
    os.system("python \"render_segment_main.py\" --server_config=3 --num_workers=4 --img_to_load=-1 --load_previous=0 --test_mode=0 --batch_size=1024 "
              "--net_config=2 --l1_weight=10.0 --use_bce=0 --min_epochs=120 "
              "--version_name=\"rgb2smoothness_v1.08\" --iteration=1 --map_choice=\"smoothness\"")

if __name__ == "__main__":
    main()