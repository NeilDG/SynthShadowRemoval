import os
def main():
    os.system("python \"transfer_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=512 --num_blocks=0 --likeness_weight=1.0 --use_bce=1 --weather=\"night\"")

if __name__ == "__main__":
    main()