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
        self.epoch_map = {"train_style_transfer" : 0, "train_shadow_mask" : 0, "train_shadow" : 0, "train_shadow_refine" : 0}

        # COARE, CCS CLOUD, GCLOUD, RTX 2080TI, RTX 3090
        if(constants.server_config <= 5):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 5, "max_epochs" : 25},
                                    "train_shadow_mask": {"min_epochs": 2, "max_epochs" : 15, "patch_size": 128},
                                    "train_shadow": {"min_epochs": 20 ,"max_epochs" : 80, "patch_size": 128},
                                    "train_shadow_refine": {"min_epochs": 30,"max_epochs" : 80, "patch_size": 128}}
        #debug
        if(constants.debug_run == 1):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 1, "max_epochs" : 5},
                                    "train_shadow_mask": {"min_epochs": 1, "max_epochs" : 15, "patch_size": 128},
                                    "train_shadow": {"min_epochs": 1,"max_epochs" : 40, "patch_size": 128},
                                    "train_shadow_refine": {"min_epochs": 10,"max_epochs" : 30, "patch_size": 128}}


        self.update_version_config()


    def update_version_config(self):
        self.version_config = {"version": constants.network_version, "network_p_name": "rgb2shadowmask", "network_z_name": "rgb2ns",
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
        NUM_BLOCKS_KEY = "num_blocks"
        LOAD_SIZE_KEY_Z = "load_size_z"
        BATCH_SIZE_KEY_Z = "batch_size_z"
        LOAD_SIZE_KEY_P = "load_size_p"
        BATCH_SIZE_KEY_P = "batch_size_p"
        SYNTH_DATASET_VERSION = "dataset_version"
        WEIGHT_DECAY_KEY = "weight_decay"
        DROPOUT_KEY = "use_dropout"
        AUGMENT_KEY = "augment_mode"

        TRAIN_MODE_KEY = "train_mode" #0 = end2end, 1 = end2end with masking, 2 = by shadow map, 3 = end2end. mask + input concat
        # 4 = by shadow map. mask + input concat

        #set defaults
        network_config[NETWORK_CONFIG_NUM] = 5
        network_config[NC_KEY] = 3
        network_config[NUM_BLOCKS_KEY] = 3
        network_config[SYNTH_DATASET_VERSION] = "v17"
        network_config[WEIGHT_DECAY_KEY] = 0.0
        network_config[DROPOUT_KEY] = False
        network_config[TRAIN_MODE_KEY] = 1
        network_config[AUGMENT_KEY] = "none"

        # configure load sizes (GPU memory allocation of data) #for 128
        if (constants.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY_Z] = 64
        elif (constants.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY_Z] = 96
        elif (constants.server_config == 3):  # GCLOUD
            network_config[LOAD_SIZE_KEY_Z] = 16
        elif (constants.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY_Z] = 32
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY_Z] = 64

        #configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        # configure load sizes (GPU memory allocation of data) #for 128
        if (constants.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY_P] = 96
        elif (constants.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY_P] = 256
        elif (constants.server_config == 3):  # GCLOUD
            network_config[LOAD_SIZE_KEY_P] = 96
        elif (constants.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY_P] = 64
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY_P] = 128

        # configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY_P] = network_config[LOAD_SIZE_KEY_P]

        if(constants.network_version == "v51.01"):
            network_config[SYNTH_DATASET_VERSION] = "v17"

        elif (constants.network_version == "v52.01"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[TRAIN_MODE_KEY] = 3
            network_config[NC_KEY] = 4 #RGB image + mask image
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NUM_BLOCKS_KEY] = 3

        elif (constants.network_version == "v53.01"):
            network_config[SYNTH_DATASET_VERSION] = "v14"

        elif (constants.network_version == "v53.02"):
            network_config[SYNTH_DATASET_VERSION] = "v14"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01

        elif (constants.network_version == "v53.03"):
            network_config[SYNTH_DATASET_VERSION] = "v14"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0

        elif (constants.network_version == "v53.04"):
            network_config[SYNTH_DATASET_VERSION] = "v14"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01

        elif (constants.network_version == "v53.05"):
            network_config[SYNTH_DATASET_VERSION] = "v14"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0

        elif (constants.network_version == "v54.02"):
            network_config[SYNTH_DATASET_VERSION] = "v8"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01

        elif (constants.network_version == "v54.03"):
            network_config[SYNTH_DATASET_VERSION] = "v8"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0

        elif (constants.network_version == "v54.04"):
            network_config[SYNTH_DATASET_VERSION] = "v8"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01

        elif (constants.network_version == "v54.05"):
            network_config[SYNTH_DATASET_VERSION] = "v8"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0

        elif (constants.network_version == "v55.02"):
            network_config[SYNTH_DATASET_VERSION] = "v18"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01

        elif (constants.network_version == "v55.03"):
            network_config[SYNTH_DATASET_VERSION] = "v18"
            network_config[AUGMENT_KEY] = "augmix"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0

        elif (constants.network_version == "v55.04"):
            network_config[SYNTH_DATASET_VERSION] = "v18"
            network_config[TRAIN_MODE_KEY] = 2
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            # network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[NETWORK_CONFIG_NUM] = 4

            network_config[LOAD_SIZE_KEY_Z] = 48
            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (constants.network_version == "v55.05"):
            network_config[SYNTH_DATASET_VERSION] = "v18"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = False
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NUM_BLOCKS_KEY] = 3

        elif (constants.network_version == "v56.04"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

        elif (constants.network_version == "v56.06"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            # network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

            # configure load sizes (GPU memory allocation of data) #for 128
            if (constants.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (constants.server_config == 3):  # GCLOUD
                network_config[LOAD_SIZE_KEY_Z] = 16
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (constants.network_version == "v56.07"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[TRAIN_MODE_KEY] = 3
            network_config[NETWORK_CONFIG_NUM] = 6
            network_config[DROPOUT_KEY] = True
            network_config[WEIGHT_DECAY_KEY] = 0.01
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

        elif (constants.network_version == "v56.08"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[AUGMENT_KEY] = "trivial_augment_wide"
            network_config[TRAIN_MODE_KEY] = 3
            network_config[NETWORK_CONFIG_NUM] = 6
            network_config[WEIGHT_DECAY_KEY] = 0.001
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

        elif (constants.network_version == "v56.09"):
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[TRAIN_MODE_KEY] = 3
            network_config[NETWORK_CONFIG_NUM] = 6
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

        elif (constants.network_version == "v56.10"): #from v56.06. Just use dropout this time
            network_config[SYNTH_DATASET_VERSION] = "v17"
            network_config[DROPOUT_KEY] = True
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NUM_BLOCKS_KEY] = 6
            network_config[LOAD_SIZE_KEY_Z] = 32

            # configure load sizes (GPU memory allocation of data) #for 128
            if (constants.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (constants.server_config == 3):  # GCLOUD
                network_config[LOAD_SIZE_KEY_Z] = 16
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]



        return network_config

    def interpret_style_transfer_config_from_version(self):
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY = "batch_size"
        PATCH_SIZE_KEY = "patch_size"
        LOAD_SIZE_KEY = "load_size"
        NORM_MODE_KEY = "norm_mode"

        #set defaults
        network_config[PATCH_SIZE_KEY] = 32
        if (constants.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY] = 16
        elif (constants.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY] = 16
        elif (constants.server_config == 3):  # GCLOUD
            network_config[LOAD_SIZE_KEY] = 16
        elif (constants.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY] = 8
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY] = 16

        # configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY] = 256
        network_config[NORM_MODE_KEY] = "batch"

        if(constants.network_version == "v8.03"): #AdainGEN
            network_config[NETWORK_CONFIG_NUM] = 3
            network_config[NUM_BLOCKS_KEY] = 4

            if (constants.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY] = 8
            elif (constants.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY] = 8
            elif (constants.server_config == 3):  # GCLOUD
                network_config[LOAD_SIZE_KEY] = 8
            elif (constants.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY] = 8

        elif(constants.network_version == "v8.04"): #Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0

        elif (constants.network_version == "v8.05"):  # Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[NORM_MODE_KEY] = "instance"

        elif (constants.network_version == "v8.06"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "batch"

        elif (constants.network_version == "v8.07"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "instance"
        else:
            print("Network config not found for ", constants.network_version)

        return network_config


