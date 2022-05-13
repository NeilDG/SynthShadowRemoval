from enum import Enum
class IterationTable():
    class NetworkType(Enum):
        ALBEDO = "ALBEDO",
        SHADING = "SHADING",
        SHADOW = "SHADOW"

    def __init__(self):
        #initialize table
        self.iteration_table = {}

        iteration = 5
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, l1_weight=0.0, lpip_weight=0.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADOW)] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)

        iteration = 6
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, l1_weight=0.0, lpip_weight=0.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADOW)] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)

        iteration = 7
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, l1_weight=0.0, lpip_weight=0.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADOW)] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, bce_weight=0.0, is_bce=0)

        iteration = 8
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, l1_weight=0.0, lpip_weight=0.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADOW)] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, bce_weight=0.0, is_bce=1)

    def get_version(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_version()

    def get_l1_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_l1_weight()

    def get_lpip_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_lpip_weight()

    def get_ssim_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_ssim_weight()

    def get_bce_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_bce_weight()

    def is_bce_enabled(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].is_bce_enabled()

class IterationParameters():
    def __init__(self, iteration, l1_weight, lpip_weight, ssim_weight, bce_weight, is_bce):
        self.iteration = iteration
        self.l1_weight = l1_weight
        self.lpip_weight = lpip_weight
        self.ssim_weight = ssim_weight
        self.bce_weight = bce_weight
        self.is_bce = is_bce

    def get_version(self):
        return self.iteration

    def get_l1_weight(self):
        return self.l1_weight

    def get_lpip_weight(self):
        return self.lpip_weight

    def get_ssim_weight(self):
        return self.ssim_weight

    def get_bce_weight(self):
        return self.bce_weight

    def is_bce_enabled(self):
        return self.is_bce