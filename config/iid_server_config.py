import constants

class IIDServerConfig():
    _sharedInstance = None

    @staticmethod
    def initialize(version):
        if(IIDServerConfig._sharedInstance == None):
            IIDServerConfig._sharedInstance = IIDServerConfig(version)

    @staticmethod
    def getInstance():
        return IIDServerConfig._sharedInstance

    def __init__(self, version):
        self.epoch_map = {"train_style_transfer" : 0, "train_albedo_mask" : 0, "train_albedo" : 0, "train_shading" : 0, "train_shadow" : 0}

        # COARE, CCS CLOUD, GCLOUD, RTX 2080TI, RTX 3090
        if(constants.server_config <= 5):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 20, "max_epochs" : 100},
                                    "train_albedo_mask": {"min_epochs": 3, "max_epochs" : 10, "patch_size": 256},
                                    "train_albedo": {"min_epochs": 10,"max_epochs" : 40, "patch_size": 64},
                                    "train_shading": {"min_epochs": 10,"max_epochs" : 40, "patch_size": 64},
                                    "train_shadow": {"min_epochs": 10,"max_epochs" : 40, "patch_size": 64}}
        #debug
        if(constants.debug_run == 1):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 1, "max_epochs" : 5},
                                    "train_albedo_mask": {"min_epochs": 1, "max_epochs" : 2, "patch_size": 256},
                                   "train_albedo": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 64},
                                   "train_shading": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 64},
                                    "train_shadow": {"min_epochs": 1,"max_epochs" : 2, "patch_size": 64}}


        self.version_config = {"version": version, "network_p_name": "rgb2mask", "network_a_name" : "rgb2albedo", "network_s_name" : "rgb2shading", "network_z_name" : "rgb2noshadow",
                               "style_transfer_name": "synth2rgb"}


    def update_version_config(self, version):
        self.version_config = {"version": version, "network_p_name": "rgb2mask", "network_a_name": "rgb2albedo", "network_s_name": "rgb2shading", "network_z_name": "rgb2noshadow",
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

    def interpret_network_config_from_version(self, version): #interprets a given version name + iteration, to its corresponding network config. Ex: v.9.00.XX = U-Net config
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NC_KEY = "nc"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY_P = "batch_size_p"
        BATCH_SIZE_KEY_A = "batch_size_a"
        BATCH_SIZE_KEY_S = "batch_size_s"
        BATCH_SIZE_KEY_Z = "batch_size_z"
        ALBEDO_MODE_KEY = "albedo_mode"
        DA_ENABLED = "da_enabled"
        STYLE_TRANSFER = "style_transferred"
        ISTD_MIX_KEY = "istd_mix"

        network_config["unlit_version_name"] = "synth2unlit_v1.00_1.pt"
        network_config["da_version_name"] = "embedding_v5.00_5"

        if (version == "v17.07" or version == "v14.07" or version == "v13.07" or version == "v12.07"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[DA_ENABLED] = 0
            network_config[STYLE_TRANSFER] = 0

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 16
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 64
                network_config[BATCH_SIZE_KEY_S] = 64
                network_config[BATCH_SIZE_KEY_Z] = 128
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256

        elif (version == "v17.10" or version == "v14.10" or version == "v13.10" or version == "v12.10"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[DA_ENABLED] = 0
            network_config[STYLE_TRANSFER] = 1
            network_config[ISTD_MIX_KEY] = True

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 16
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 64
                network_config[BATCH_SIZE_KEY_S] = 64
                network_config[BATCH_SIZE_KEY_Z] = 64
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256

        elif (version == "v17.11"):  # Adain-GEN
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NC_KEY] = 3
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[ALBEDO_MODE_KEY] = 1
            network_config[DA_ENABLED] = 0
            network_config[STYLE_TRANSFER] = 1
            network_config[ISTD_MIX_KEY] = False

            # configure batch sizes
            if (constants.server_config == 1):  # COARE
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 16
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[BATCH_SIZE_KEY_P] = 24
                network_config[BATCH_SIZE_KEY_A] = 192
                network_config[BATCH_SIZE_KEY_S] = 192
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 3):  # GCLOUD
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[BATCH_SIZE_KEY_P] = 8
                network_config[BATCH_SIZE_KEY_A] = 64
                network_config[BATCH_SIZE_KEY_S] = 64
                network_config[BATCH_SIZE_KEY_Z] = 64
            else:  # RTX 3090
                network_config[BATCH_SIZE_KEY_P] = 16
                network_config[BATCH_SIZE_KEY_A] = 128
                network_config[BATCH_SIZE_KEY_S] = 128
                network_config[BATCH_SIZE_KEY_Z] = 256

        return network_config

    def interpret_style_transfer_config_from_version(self, version):
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY = "batch_size"
        PATCH_SIZE_KEY = "patch_size"
        IMG_PER_ITER = "img_per_iter"

        if(version == "v6.03"): #AdainGEN
            network_config[NETWORK_CONFIG_NUM] = 3
            network_config[NUM_BLOCKS_KEY] = 4
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 512
            network_config[IMG_PER_ITER] = 8
        elif(version == "v6.04"): #Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[PATCH_SIZE_KEY] = 32
            network_config[BATCH_SIZE_KEY] = 1024
            network_config[IMG_PER_ITER] = 32
        else:
            print("Network config not found for ", version)

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
        else:
            print("Mode ", mode, " not recognized.")
            return -1

