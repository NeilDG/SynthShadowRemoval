import global_config

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
        self.epoch_map = {"train_style_transfer" : 0, "train_shadow_matte" : 0, "train_shadow" : 0}

        # COARE, CCS CLOUD, GCLOUD, RTX 2080TI, RTX 3090
        if(global_config.server_config <= 5):
            self.general_configs = {"train_style_transfer" : {"min_epochs" : 5, "max_epochs" : 25},
                                    "train_shadow_matte": {"min_epochs": 5, "max_epochs" : 15},
                                    "train_shadow": {"min_epochs": 10 ,"max_epochs" : 80}}
        #debug
        if(global_config.debug_run == 1):
            self.general_configs = {"train_style_transfer": {"min_epochs": 1, "max_epochs": 25},
                                    "train_shadow_matte": {"min_epochs": 1, "max_epochs": 15},
                                    "train_shadow": {"min_epochs": 1, "max_epochs": 80}}


        self.update_version_config()


    def update_version_config(self):
        self.version_config = {"shadow_network_version": global_config.shadow_removal_version, "shadow_matte_network_version": global_config.shadow_matte_network_version,
                               "style_transfer_version" : global_config.style_transfer_version,
                               "network_m_name": "rgb2sm", "network_z_name": "rgb2ns", "style_transfer_name": "synth2rgb"}

    def get_general_configs(self):
        return self.general_configs

    def get_version_config(self, version_keyword, network_name, iteration):
        network = self.version_config[network_name]
        version = self.version_config[version_keyword]

        return network + "_" + version + "_" + str(iteration)

    def store_epoch_from_checkpt(self, mode, epoch):
        self.epoch_map[mode] = epoch

    def get_last_epoch_from_mode(self, mode):
        return self.epoch_map[mode]


    def interpret_shadow_matte_params_from_version(self):  # interprets a given version name + iteration, to its corresponding network config.
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NC_KEY = "nc"
        NUM_BLOCKS_KEY = "num_blocks"
        LOAD_SIZE_KEY_M = "load_size_m"
        BATCH_SIZE_KEY_M = "batch_size_m"
        SYNTH_DATASET_VERSION = "dataset_version"
        WEIGHT_DECAY_KEY = "weight_decay"
        DROPOUT_RATE_KEY = "dropout_rate"
        AUGMENT_KEY = "augment_mode"
        DATASET_REPEAT_KEY = "dataset_repeats"
        MIX_ISTD_KEY = "mix_istd"
        USE_ISTD_POOL_KEY = "use_istd_pool"
        PATCH_SIZE_KEY = "patch_size"

        # set defaults
        network_config[NETWORK_CONFIG_NUM] = 5
        network_config[NC_KEY] = 3
        network_config[NUM_BLOCKS_KEY] = 3
        network_config[PATCH_SIZE_KEY] = 64
        network_config[SYNTH_DATASET_VERSION] = "v32_istd_styled_1"
        network_config[WEIGHT_DECAY_KEY] = 0.005
        network_config[DROPOUT_RATE_KEY] = 0.5
        network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]
        network_config[DATASET_REPEAT_KEY] = 1
        network_config[MIX_ISTD_KEY] = 0.0 # percent to use ISTD
        network_config[USE_ISTD_POOL_KEY] = False

        # configure load sizes (GPU memory allocation of data) #for 128
        if (global_config.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY_M] = 128
        elif (global_config.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY_M] = 128
        elif (global_config.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY_M] = 64
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY_M] = 128

        # configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        assert "v58.28" in global_config.shadow_matte_network_version or "v58.39" in global_config.shadow_matte_network_version \
               or "v58.65" in global_config.shadow_matte_network_version or "v60" in global_config.shadow_matte_network_version, "Shadow matte network version not recognized: " + global_config.shadow_matte_network_version

        #TODO: Temporary - for quick experiment. K dataset repeats to lessen number of epochs, given <2000 images
        network_config[DATASET_REPEAT_KEY] = 30
        self.general_configs["train_shadow_matte"]["min_epochs"] = 60
        self.general_configs["train_shadow_matte"]["max_epochs"] = 65

        if (global_config.shadow_matte_network_version == "v58.28" or global_config.shadow_matte_network_version == "v58.39"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            network_config[NUM_BLOCKS_KEY] = 15


        elif (global_config.shadow_matte_network_version == "v58.65"):
            network_config[SYNTH_DATASET_VERSION] = "v32_istd"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 16
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.05"):
            network_config[SYNTH_DATASET_VERSION] = "v32_istd_styled_1"
            network_config[PATCH_SIZE_KEY] = 64
        elif (global_config.shadow_matte_network_version == "v60.06"):
            network_config[SYNTH_DATASET_VERSION] = "v32_istd_styled_2"
            network_config[PATCH_SIZE_KEY] = 64

        elif (global_config.shadow_matte_network_version == "v60.07"):
            network_config[SYNTH_DATASET_VERSION] = "v32_istd_styled_1"
            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]
        elif (global_config.shadow_matte_network_version == "v60.08"):
            network_config[SYNTH_DATASET_VERSION] = "v32_istd_styled_2"
            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

        elif (global_config.shadow_matte_network_version == "v60.09"):
            network_config[SYNTH_DATASET_VERSION] = "v34_places"
            network_config[PATCH_SIZE_KEY] = 64

        elif (global_config.shadow_matte_network_version == "v60.10"):
            network_config[SYNTH_DATASET_VERSION] = "v35_places"
            network_config[PATCH_SIZE_KEY] = 64

        elif (global_config.shadow_matte_network_version == "v60.11_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[PATCH_SIZE_KEY] = 64
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = []
            network_config[DATASET_REPEAT_KEY] = 120

        elif (global_config.shadow_matte_network_version == "v60.12_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[PATCH_SIZE_KEY] = 64
            network_config[DATASET_REPEAT_KEY] = 120

        elif (global_config.shadow_matte_network_version == "v60.13_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = []
            network_config[DATASET_REPEAT_KEY] = 120

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.14_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[DATASET_REPEAT_KEY] = 120
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.15_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[DATASET_REPEAT_KEY] = 120
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 32
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 48
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 16
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 32

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.16_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = []
            network_config[DATASET_REPEAT_KEY] = 120
            network_config[NUM_BLOCKS_KEY] = 6

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 12

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

            self.general_configs["train_shadow_matte"]["min_epochs"] = 65
            self.general_configs["train_shadow_matte"]["max_epochs"] = 65

        elif (global_config.shadow_matte_network_version == "v60.17_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = []
            network_config[DATASET_REPEAT_KEY] = 120
            network_config[NUM_BLOCKS_KEY] = 9

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 8

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]
            self.general_configs["train_shadow_matte"]["min_epochs"] = 45
            self.general_configs["train_shadow_matte"]["max_epochs"] = 50

        elif (global_config.shadow_matte_network_version == "v60.15_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[PATCH_SIZE_KEY] = 64
            network_config[WEIGHT_DECAY_KEY] = 0.0

            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

        elif (global_config.shadow_matte_network_version == "v60.16_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

        elif (global_config.shadow_matte_network_version == "v60.17_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[PATCH_SIZE_KEY] = 64
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10


        elif (global_config.shadow_matte_network_version == "v60.18_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.19_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 6
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 12
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 12
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 12
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.20_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v1_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.21_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v2_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.22_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v3_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 64
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.23_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 64
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.24_places"):
            network_config[SYNTH_DATASET_VERSION] = "v46_places"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.25_places"):
            network_config[SYNTH_DATASET_VERSION] = "v49_places"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.26_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v4_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.27_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v5_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.28_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v6_synshadow"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.29_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = []
            network_config[DATASET_REPEAT_KEY] = 120

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.30_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[DATASET_REPEAT_KEY] = 120

            network_config[PATCH_SIZE_KEY] = 256
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.31_srd"): #retrain SRD
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            self.general_configs["train_shadow_matte"]["patch_size"] = 256
            network_config[AUGMENT_KEY] = ["random_noise", "random_exposure"]

            network_config[DATASET_REPEAT_KEY] = 120
            self.general_configs["train_shadow_matte"]["min_epochs"] = 5
            self.general_configs["train_shadow_matte"]["max_epochs"] = 10

            network_config[NUM_BLOCKS_KEY] = 3
            network_config[LOAD_SIZE_KEY_M] = 128
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.31_places"):
            network_config[SYNTH_DATASET_VERSION] = "v50_places"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 20
            self.general_configs["train_shadow_matte"]["max_epochs"] = 25

            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 16
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 16
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.32_places"):
            network_config[SYNTH_DATASET_VERSION] = "v50_places"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 20
            self.general_configs["train_shadow_matte"]["max_epochs"] = 25

            network_config[NUM_BLOCKS_KEY] = 6
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 12
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 12
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 8
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 12
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.33_places"):
            network_config[SYNTH_DATASET_VERSION] = "v50_places"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0

            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow_matte"]["min_epochs"] = 2
            self.general_configs["train_shadow_matte"]["max_epochs"] = 3

            network_config[NUM_BLOCKS_KEY] = 9
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 8
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.34_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[DATASET_REPEAT_KEY] = 10

            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 1

            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 64
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        elif (global_config.shadow_matte_network_version == "v60.34_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[PATCH_SIZE_KEY] = 256
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]
            network_config[WEIGHT_DECAY_KEY] = 0.0
            network_config[DROPOUT_RATE_KEY] = 0.0
            network_config[DATASET_REPEAT_KEY] = 10

            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 1

            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_M] = 8
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_M] = 4
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_M] = 64
            network_config[BATCH_SIZE_KEY_M] = network_config[LOAD_SIZE_KEY_M]

        return network_config

    def interpret_shadow_network_params_from_version(self): #interprets a given version name + iteration, to its corresponding network config.
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        NC_KEY = "nc"
        PATCH_SIZE_KEY = "patch_size"
        NUM_BLOCKS_KEY = "num_blocks"
        LOAD_SIZE_KEY_Z = "load_size_z"
        BATCH_SIZE_KEY_Z = "batch_size_z"
        SYNTH_DATASET_VERSION = "dataset_version"
        WEIGHT_DECAY_KEY = "weight_decay"
        DROPOUT_RATE_KEY = "dropout_rate"
        AUGMENT_KEY = "augment_mode"
        DATASET_REPEAT_KEY = "dataset_repeats"
        MIX_ISTD_KEY = "mix_istd"

        #set defaults
        network_config[PATCH_SIZE_KEY] = 64
        network_config[NETWORK_CONFIG_NUM] = 6
        network_config[NC_KEY] = 3
        network_config[NUM_BLOCKS_KEY] = 3
        network_config[SYNTH_DATASET_VERSION] = "v34_places"
        network_config[WEIGHT_DECAY_KEY] = 0.0
        network_config[DROPOUT_RATE_KEY] = 0.0
        network_config[AUGMENT_KEY] = "none"
        network_config[DATASET_REPEAT_KEY] = 1
        network_config[MIX_ISTD_KEY] = 0.0  # percent to use ISTD

        # configure load sizes (GPU memory allocation of data) #for 128
        if (global_config.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY_Z] = 128
        elif (global_config.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY_Z] = 128
        elif (global_config.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY_Z] = 64
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY_Z] = 96

        #configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]


        assert "v58" in global_config.shadow_removal_version or "v60" in global_config.shadow_removal_version, "Shadow network version not recognized: " + global_config.shadow_removal_version

        # TODO: Temporary - for quick experiment. K dataset repeats to lessen number of epochs, given <2000 images
        network_config[DATASET_REPEAT_KEY] = 30
        self.general_configs["train_shadow"]["min_epochs"] = 5
        self.general_configs["train_shadow"]["max_epochs"] = 10


        if (global_config.shadow_removal_version == "v60.01_places"):
            network_config[SYNTH_DATASET_VERSION] = "v34_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10

        elif (global_config.shadow_removal_version == "v60.01_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v34_istd"

        elif (global_config.shadow_removal_version == "v60.01_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v34_srd"

            self.general_configs["train_shadow"]["min_epochs"] = 15
            self.general_configs["train_shadow"]["max_epochs"] = 20

        elif (global_config.shadow_removal_version == "v60.02_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v35_istd"

        elif(global_config.shadow_removal_version == "v60.02_places"):
            network_config[SYNTH_DATASET_VERSION] = "v35_places"

        elif (global_config.shadow_removal_version == "v60.03_places"):
            network_config[SYNTH_DATASET_VERSION] = "v34_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.04_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.05_places"):
            network_config[SYNTH_DATASET_VERSION] = "v36_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.06_places"):
            network_config[SYNTH_DATASET_VERSION] = "v37_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.07_places"):
            network_config[SYNTH_DATASET_VERSION] = "v37_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.08_places"):
            network_config[SYNTH_DATASET_VERSION] = "v38_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.09_places"):
            network_config[SYNTH_DATASET_VERSION] = "v38_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.10_places"):
            network_config[SYNTH_DATASET_VERSION] = "v42_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 15
            self.general_configs["train_shadow"]["max_epochs"] = 20
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.11_places"):
            network_config[SYNTH_DATASET_VERSION] = "v46_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.12_places"):
            network_config[SYNTH_DATASET_VERSION] = "v43_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.13_places"):
            network_config[SYNTH_DATASET_VERSION] = "v48_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.14_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v1_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.15_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v2_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.16_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v3_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.17_places"):
            network_config[SYNTH_DATASET_VERSION] = "v50_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.18_places"):
            network_config[SYNTH_DATASET_VERSION] = "v44_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.19_places"):
            network_config[SYNTH_DATASET_VERSION] = "v51_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.20_places"):
            network_config[SYNTH_DATASET_VERSION] = "v47_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.21_places"):
            network_config[SYNTH_DATASET_VERSION] = "v45_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.22_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v1_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.23_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v2_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.24_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v3_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.25_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v4_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.26_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v5_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.27_synshadow"):
            network_config[SYNTH_DATASET_VERSION] = "v6_synshadow"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.28_places"):
            network_config[SYNTH_DATASET_VERSION] = "v49_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.29_istd"):
            network_config[SYNTH_DATASET_VERSION] = "v_istd"
            network_config[DATASET_REPEAT_KEY] = 3
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.30_srd"):
            network_config[SYNTH_DATASET_VERSION] = "v_srd"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.31_places"):
            network_config[SYNTH_DATASET_VERSION] = "v39_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128
            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.32_places"):
            network_config[SYNTH_DATASET_VERSION] = "v46_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128

            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 1
            network_config[NC_KEY] = 4

            # configure load sizes (GPU memory allocation of data) #for 128
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 256

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]

        elif (global_config.shadow_removal_version == "v60.33_places"):
            network_config[SYNTH_DATASET_VERSION] = "v52_places"
            network_config[DATASET_REPEAT_KEY] = 10
            self.general_configs["train_shadow"]["min_epochs"] = 5
            self.general_configs["train_shadow"]["max_epochs"] = 10
            network_config[AUGMENT_KEY] = ["augmix", "random_noise", "random_exposure"]

            network_config[PATCH_SIZE_KEY] = 128

            # configure load sizes (GPU memory allocation of data)
            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY_Z] = 64
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY_Z] = 96
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY_Z] = 32
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY_Z] = 64

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY_Z] = network_config[LOAD_SIZE_KEY_Z]


        return network_config

    def interpret_style_transfer_config_from_version(self):
        network_config = {}
        NETWORK_CONFIG_NUM = "net_config"
        SYNTH_DATASET_VERSION = "dataset_version"
        NUM_BLOCKS_KEY = "num_blocks"
        BATCH_SIZE_KEY = "batch_size"
        PATCH_SIZE_KEY = "patch_size"
        LOAD_SIZE_KEY = "load_size"
        NORM_MODE_KEY = "norm_mode"
        DATASET_REPEAT_KEY = "dataset_repeats"

        #set defaults
        network_config[PATCH_SIZE_KEY] = 32
        if (global_config.server_config == 1):  # COARE
            network_config[LOAD_SIZE_KEY] = 512
        elif (global_config.server_config == 2):  # CCS JUPYTER
            network_config[LOAD_SIZE_KEY] = 512
        elif (global_config.server_config == 4):  # RTX 2080Ti
            network_config[LOAD_SIZE_KEY] = 256
        else:  # RTX 3090
            network_config[LOAD_SIZE_KEY] = 512

        # configure batch size. NOTE: Batch size must be equal or larger than load size
        network_config[BATCH_SIZE_KEY] = network_config[LOAD_SIZE_KEY]
        network_config[NORM_MODE_KEY] = "batch"
        network_config[SYNTH_DATASET_VERSION] = "v32_istd"

        # TODO: Temporary - for quick experiment. K dataset repeats to lessen number of epochs, given <2000 images
        network_config[DATASET_REPEAT_KEY] = 40
        self.general_configs["train_style_transfer"]["min_epochs"] = 15
        self.general_configs["train_style_transfer"]["max_epochs"] = 20

        assert "v10" in global_config.style_transfer_version, "Style transfer network version not recognized: " + global_config.style_transfer_version

        if(global_config.style_transfer_version == "v10.03"): #AdainGEN
            network_config[NETWORK_CONFIG_NUM] = 3
            network_config[NUM_BLOCKS_KEY] = 4

            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY] = 128
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY] = 576
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY] = 64
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY] = 128

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY] = network_config[LOAD_SIZE_KEY]

        elif(global_config.style_transfer_version == "v10.04"): #Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0

        elif (global_config.style_transfer_version == "v10.05"):  # Unet
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[NORM_MODE_KEY] = "instance"

        elif (global_config.style_transfer_version == "v10.06"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "batch"

        elif (global_config.style_transfer_version == "v10.07"):  # Cycle-GAN
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "instance"

        elif (global_config.style_transfer_version == "v10.08"):  # Cycle-GAN
            network_config[PATCH_SIZE_KEY] = 64
            network_config[NETWORK_CONFIG_NUM] = 1
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "instance"

            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY] = 128
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY] = 340
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY] = 64
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY] = 128

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY] = network_config[LOAD_SIZE_KEY]

        elif (global_config.style_transfer_version == "v10.09"):  # U-Net
            network_config[PATCH_SIZE_KEY] = 64
            network_config[NETWORK_CONFIG_NUM] = 2
            network_config[NUM_BLOCKS_KEY] = 0
            network_config[NORM_MODE_KEY] = "instance"

            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY] = 512
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY] = 512
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY] = 256
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY] = 512

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY] = network_config[LOAD_SIZE_KEY]

        elif (global_config.style_transfer_version == "v10.10"):  # Cycle-GAN-CBAM
            network_config[PATCH_SIZE_KEY] = 64
            network_config[NETWORK_CONFIG_NUM] = 4
            network_config[NUM_BLOCKS_KEY] = 10
            network_config[NORM_MODE_KEY] = "instance"

            if (global_config.server_config == 1):  # COARE
                network_config[LOAD_SIZE_KEY] = 128
            elif (global_config.server_config == 2):  # CCS JUPYTER
                network_config[LOAD_SIZE_KEY] = 340
            elif (global_config.server_config == 4):  # RTX 2080Ti
                network_config[LOAD_SIZE_KEY] = 64
            else:  # RTX 3090
                network_config[LOAD_SIZE_KEY] = 128

            # configure batch size. NOTE: Batch size must be equal or larger than load size
            network_config[BATCH_SIZE_KEY] = network_config[LOAD_SIZE_KEY]

        else:
            print("Network config not found for ", global_config.style_transfer_version)

        return network_config


