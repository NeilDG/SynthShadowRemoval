import constants

class IIDServerConfig():
    _sharedInstance = None

    @staticmethod
    def initialize():
        if(IIDServerConfig._sharedInstance == None):
            IIDServerConfig._sharedInstance = IIDServerConfig()

    @staticmethod
    def getInstance():
        return IIDServerConfig._sharedInstance

    def __init__(self):
        self.epoch_map = {"train_style_transfer" : 0, "train_albedo_mask" : 0, "train_albedo" : 0, "train_shading" : 0, "train_shadow" : 0, "train_shadow_refine" : 0}

        # COARE, CCS CLOUD, GCLOUD, RTX 2080TI, RTX 3090
        if(constants.server_config <= 5):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 5, "max_epochs" : 25},
                                    "train_albedo_mask": {"min_epochs": 3, "max_epochs" : 10, "patch_size": 256},
                                    "train_albedo": {"min_epochs": 10,"max_epochs" : 40, "patch_size": 64},
                                    "train_shading": {"min_epochs": 10,"max_epochs" : 40, "patch_size": 64},
                                    "train_shadow": {"min_epochs": 80,"max_epochs" : 120, "patch_size": 128},
                                    "train_shadow_refine": {"min_epochs": 10,"max_epochs" : 30, "patch_size": 128}}
        #debug
        if(constants.debug_run == 1):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 1, "max_epochs" : 5},
                                    "train_albedo_mask": {"min_epochs": 1, "max_epochs" : 2, "patch_size": 256},
                                   "train_albedo": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 64},
                                   "train_shading": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 64},
                                    "train_shadow": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 128},
                                    "train_shadow_refine": {"min_epochs": 10,"max_epochs" : 30, "patch_size": 128}}


        self.update_version_config()


    def update_version_config(self):
        self.version_config = {"version": constants.network_version, "network_p_name": "rgb2mask", "network_a_name": "rgb2albedo", "network_s_name": "rgb2shading", "network_z_name": "rgb2ns",
                               "network_zr_name": "rgb2ns_refine",
                               "style_transfer_name": "synth2rgb"}

    def get_general_configs(self):
        return self.general_configs

    def get_version_config(self, network_name, iteration):
        network = self.version_config[network_name]
        version = self.version_config["version"]

        return network + "_" + version + "_" + str(iteration)

    def store_epoch_from_checkpt(self, mode, epoch):
        self.epoch_map[mode] = epoch

    def get_last_epoch_from_mode(self, mode):
        return self.epoch_map[mode]

    def interpret_network_config_from_version(self): #interprets a given version name + iteration, to its corresponding network config.
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NC_KEY = "nc"
        SHADOW_REFINE_NC_KEY = "shadowrefine_nc"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY_P = "batch_size_p"
        BATCH_SIZE_KEY_A = "batch_size_a"
        BATCH_SIZE_KEY_S = "batch_size_s"
        BATCH_SIZE_KEY_Z = "batch_size_z"
        BATCH_SIZE_KEY_ZR = "batch_size_zr"
        ALBEDO_MODE_KEY = "albedo_mode"
        STYLE_TRANSFER = "style_transferred"
        SHADOW_MAP_CHANNEL_KEY = "sm_one_channel"
        MIX_TYPE_KEY = "mix_type"

        if (constants.network_version == "v30.01"):  #Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[SHADOW_REFINE_NC_KEY] = 4
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[STYLE_TRANSFER] = 1
            network_config[SHADOW_MAP_CHANNEL_KEY] = False
            network_config[MIX_TYPE_KEY] = 2

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 96
                network_config[BATCH_SIZE_KEY_ZR] = 96

        elif (constants.network_version == "v30.02"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[SHADOW_REFINE_NC_KEY] = 4
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[STYLE_TRANSFER] = 1
            network_config[SHADOW_MAP_CHANNEL_KEY] = True
            network_config[MIX_TYPE_KEY] = 2

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64


            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 96
                network_config[BATCH_SIZE_KEY_ZR] = 96

        elif (constants.network_version == "v30.03"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[SHADOW_REFINE_NC_KEY] = 4
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[STYLE_TRANSFER] = 1
            network_config[SHADOW_MAP_CHANNEL_KEY] = False
            network_config[MIX_TYPE_KEY] = 1

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64


            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 96
                network_config[BATCH_SIZE_KEY_ZR] = 96

        elif (constants.network_version == "v30.04"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[SHADOW_REFINE_NC_KEY] = 4
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[STYLE_TRANSFER] = 1
            network_config[SHADOW_MAP_CHANNEL_KEY] = True
            network_config[MIX_TYPE_KEY] = 1

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
                network_config[BATCH_SIZE_KEY_ZR] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 32
                network_config[BATCH_SIZE_KEY_ZR] = 32
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 64
                network_config[BATCH_SIZE_KEY_ZR] = 64

        return network_config

    def interpret_style_transfer_config_from_version(self):
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY = "batch_size"
        PATCH_SIZE_KEY = "patch_size"
        IMG_PER_ITER = "img_per_iter"
        NORM_MODE_KEY = "norm_mode"

        if(constants.network_version == "v7.03"): #AdainGEN
            network_config[NETWORK_CONFIG_NUM] = 3
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 512
            network_config[IMG_PER_ITER] = 8
        elif(constants.network_version == "v7.04"): #Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 1024
            network_config[IMG_PER_ITER] = 16
            network_config[NORM_MODE_KEY] = "batch"

            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 48
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 16
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 32
        elif (constants.network_version == "v7.05"):  # Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 1024
            network_config[IMG_PER_ITER] = 16
            network_config[NORM_MODE_KEY] = "instance"

            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 48
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 16
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 32
        elif (constants.network_version == "v7.06"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 1024
            network_config[IMG_PER_ITER] = 16
            network_config[NORM_MODE_KEY] = "batch"
            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 48
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 16
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 32
        elif (constants.network_version == "v7.07"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 1024
            network_config[IMG_PER_ITER] = 16
            network_config[NORM_MODE_KEY] = "instance"
            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 48
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 8
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 16

        elif (constants.network_version == "v7.10"):  # Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 256
            network_config[NORM_MODE_KEY] = "instance"

            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 56
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 16
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 32

        elif (constants.network_version == "v7.09"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 32
            network_config[NORM_MODE_KEY] = "instance"
            if (constants.server_config == 1):  # COARE
                network_config[IMG_PER_ITER] = 32
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[IMG_PER_ITER] = 48
            elif (constants.server_config == 3):  # GCLOUD
                network_config[IMG_PER_ITER] = 24
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[IMG_PER_ITER] = 8
            else:  # RTX 3090
                network_config[IMG_PER_ITER] = 16
        else:
            print("Network config not found for ", constants.network_version)

        return network_config

    def get_batch_size_from_mode(self, mode, network_config):
        if(mode == "train_albedo_mask"): #mask
            return network_config["batch_size_p"]
        elif(mode == "train_albedo"): #albedo
            return network_config["batch_size_a"]
        elif(mode == "train_shading"):
            return network_config["batch_size_s"]
        elif(mode == "train_shadow"):
            return network_config["batch_size_z"]
        elif(mode == "train_shadow_refine"):
            return network_config["batch_size_zr"]
        else:
            print("Mode ", mode, " not recognized.")
            return -1

