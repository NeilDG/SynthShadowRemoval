import constants
from config import iid_server_config
from loaders import dataset_loader
from trainers import early_stopper
from trainers.albedo_mask_trainer import AlbedoMaskTrainer
from trainers.albedo_trainer import AlbedoTrainer
from transforms import iid_transforms


class TrainerFactory():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.opts = opts

        iid_server_config.IIDServerConfig.initialize()
        self.server_config = iid_server_config.IIDServerConfig.getInstance().get_general_configs()

        self.trainer_list = {}
        self.trainer_list["train_albedo_mask"] = AlbedoMaskTrainer(self.gpu_device, opts)
        self.trainer_list["train_albedo"] = AlbedoTrainer(self.gpu_device, opts)
        # self.trainer_list["train_shading"] = AlbedoMaskTrainer(self.gpu_device, opts)

        self.iid_op = iid_transforms.IIDTransform()

    def train(self, mode, epoch, iteration, input_map, target_map):
        self.trainer_list[mode].train(epoch, iteration, input_map, target_map)


    def test(self, mode, input_map):
        self.trainer_list[mode].test(input_map)

    def is_stop_condition_met(self, mode):
        return self.trainer_list[mode].is_stop_condition_met()

    def visdom_visualize(self, mode, input_map, label = "Train"):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].visdom_visualize(input_map, label)

    def visdom_infer(self, mode, input_map):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].visdom_infer(input_map)

    def save(self, mode, epoch, iteration, is_temp:bool):
        if(self.trainer_list[mode] != None):
            self.trainer_list[mode].save_states(epoch, iteration, is_temp)

