import os
import GPUtil
import multiprocessing
import time

def train_proper(gpu_device):
    # os.system("python \"train_img2img_main.py\" --server_config=1 --cuda_device=" +gpu_device+ " --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"synth2istd_v01.00\" --iteration=5")

    os.system("python \"shadow_train_main-3.py\" --server_config=1 --cuda_device=" +gpu_device+ " --img_to_load=100000 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.26_places\" --iteration=1")

    os.system("python \"shadow_train_main-3.py\" --server_config=1 --cuda_device=" +gpu_device+ " --img_to_load=50000 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.26_places\" --iteration=1")


def main():
    EXECUTION_TIME_IN_HOURS = 48
    execution_seconds = 3600 * EXECUTION_TIME_IN_HOURS

    GPUtil.showUtilization()
    device_id = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.1, attempts=2500, interval=30, verbose=True)
    gpu_device = "cuda:" + str(device_id[0])
    print("Available GPU device found: ", gpu_device)

    p = multiprocessing.Process(target=train_proper, name="train_proper", args=(gpu_device,))
    p.start()

    time.sleep(execution_seconds) #causes p to execute code for X seconds. 3600 = 1 hour

    #terminate
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("\n Process " +p.name+ " has finished execution.")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    p.terminate()
    p.join()


if __name__ == "__main__":
    main()