import os
def main():
    os.system("python \"transfer_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=1 --batch_size=256")

if __name__ == "__main__":
    main()