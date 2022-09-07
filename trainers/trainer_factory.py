import constants
from config import iid_server_config
from loaders import dataset_loader
from model import embedding_network
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from trainers import early_stopper
from trainers.albedo_mask_trainer import AlbedoMaskTrainer
from trainers.albedo_trainer import AlbedoTrainer
from trainers.shading_trainer import ShadingTrainer
from trainers.shadow_trainer import ShadowTrainer
from transforms import iid_transforms
import torch

class TrainerFactory():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.opts = opts
        self.iid_op = iid_transforms.IIDTransform()
        self.trainer_list = {}

    def initialize_all_trainers(self, opts):
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        self.server_config = sc_instance.get_general_configs()
        self.network_config = sc_instance.interpret_network_config_from_version(opts.version)

        self.trainer_list["train_albedo"] = AlbedoTrainer(self.gpu_device, opts)
        self.trainer_list["train_shading"] = ShadingTrainer(self.gpu_device, opts)
        self.trainer_list["train_shadow"] = ShadowTrainer(self.gpu_device, opts)

        # self.initialize_da_network(self.network_config["da_version_name"])
        # self.initialize_unlit_network(self.network_config["unlit_version_name"])
        # self.trainer_list["train_albedo_mask"].assign_embedder_decoder(self.embedder, self.decoder_fixed)
        # self.trainer_list["train_albedo"].assign_embedder_decoder(self.embedder, self.decoder_fixed)
        # self.trainer_list["train_shading"].assign_embedder_decoder(self.embedder, self.decoder_fixed)
        # self.trainer_list["train_shadow"].assign_embedder_decoder(self.embedder, self.decoder_fixed)

    def get_all_trainers(self, opts):
        self.initialize_all_trainers(opts)
        return self.trainer_list["train_albedo"], self.trainer_list["train_shading"], self.trainer_list["train_shadow"]

    def get_albedo_trainer(self):
        if("train_albedo" in self.trainer_list):
            return self.trainer_list["train_albedo"]
        else:
            self.trainer_list["train_albedo"] = AlbedoTrainer(self.gpu_device, self.opts)
            return self.trainer_list["train_albedo"]

    def get_shading_trainer(self):
        if("train_shading" in self.trainer_list):
            return self.trainer_list["train_shading"]
        else:
            self.trainer_list["train_shading"] = ShadingTrainer(self.gpu_device, self.opts)
            return self.trainer_list["train_shading"]

    def get_shadow_trainer(self):
        if ("train_shadow" in self.trainer_list):
            return self.trainer_list["train_shadow"]
        else:
            self.trainer_list["train_shadow"] = ShadowTrainer(self.gpu_device, self.opts)
            return self.trainer_list["train_shadow"]

    def get_unlit_network(self):
        return self.G_unlit

    def initialize_da_network(self, da_version_name):
        self.embedder = embedding_network.EmbeddingNetworkFFA(blocks=6).to(self.gpu_device)
        checkpoint = torch.load("checkpoint/" + da_version_name + ".pt", map_location=self.gpu_device)
        self.embedder.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded embedding network: ", da_version_name)

        self.decoder_fixed = embedding_network.DecodingNetworkFFA().to(self.gpu_device)
        print("Loaded fixed decoder network")

    def initialize_unlit_network(self, unlit_version_name):
#         checkpoint = torch.load("./checkpoint/" + unlit_version_name, map_location=self.gpu_device)
#         net_config = checkpoint['net_config']
#         num_blocks = checkpoint['num_blocks']
        net_config = 2
        num_blocks = 1

        if (net_config == 1):
            self.G_unlit = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_unlit = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_unlit = ffa.FFA(gps=3, blocks=num_blocks).to(self.gpu_device)

#         self.G_unlit.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded unlit network: " + unlit_version_name)

    def train(self, mode, epoch, iteration, input_map, target_map):
        self.trainer_list[mode].train(epoch, iteration, input_map, target_map)

    def test(self, mode, input_map):
        self.trainer_list[mode].test(input_map)

    def is_stop_condition_met(self, mode):
        return self.trainer_list[mode].is_stop_condition_met()

    def visdom_plot(self, mode, iteration):
        if (self.trainer_list[mode] != None):
            self.trainer_list[mode].visdom_plot(iteration)

    def visdom_visualize(self, mode, input_map, label = "Train"):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].visdom_visualize(input_map, label)

    def visdom_infer(self, mode, input_map):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].visdom_infer(input_map)

    def save(self, mode, epoch, iteration, is_temp:bool):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].save_states(epoch, iteration, is_temp)

