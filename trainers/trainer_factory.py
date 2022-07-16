import constants
from config import iid_server_config
from loaders import dataset_loader
from model import embedding_network
from trainers import early_stopper
from trainers.albedo_mask_trainer import AlbedoMaskTrainer
from trainers.albedo_trainer import AlbedoTrainer
from trainers.shading_trainer import ShadingTrainer
from transforms import iid_transforms
import torch

class TrainerFactory():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.opts = opts

        iid_server_config.IIDServerConfig.initialize()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        self.server_config = sc_instance.get_general_configs()
        self.network_config = sc_instance.interpret_network_config_from_version(opts.version)

        self.trainer_list = {}
        self.trainer_list["train_albedo_mask"] = AlbedoMaskTrainer(self.gpu_device, opts)
        self.trainer_list["train_albedo"] = AlbedoTrainer(self.gpu_device, opts)
        self.trainer_list["train_shading"] = ShadingTrainer(self.gpu_device, opts)

        self.initialize_da_network(self.network_config["da_version_name"])
        self.trainer_list["train_albedo_mask"].assign_embedder_decoder(self.embedder, self.decoder_fixed)
        self.trainer_list["train_albedo"].assign_embedder_decoder(self.embedder, self.decoder_fixed)
        self.trainer_list["train_shading"].assign_embedder_decoder(self.embedder, self.decoder_fixed)

        self.iid_op = iid_transforms.IIDTransform()

    def initialize_da_network(self, da_version_name):
        self.embedder = embedding_network.EmbeddingNetworkFFA(blocks=6).to(self.gpu_device)
        checkpoint = torch.load("checkpoint/" + da_version_name + ".pt", map_location=self.gpu_device)
        self.embedder.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded embedding network: ", da_version_name)

        self.decoder_fixed = embedding_network.DecodingNetworkFFA().to(self.gpu_device)
        print("Loaded fixed decoder network")

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

