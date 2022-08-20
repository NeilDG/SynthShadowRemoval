from enum import Enum
class IterationTable():
    class NetworkType(Enum):
        ALBEDO = "ALBEDO",
        SHADING = "SHADING",
        SHADOW = "SHADOW"

    def __init__(self):
        #initialize table
        self.iteration_table = {}

        iteration = 1
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0], is_bce=0)

        iteration = 2
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0], is_bce=1)

        iteration = 3
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0, 1.0, 1.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0, 1.0, 1.0], is_bce=0)

        iteration = 4
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0, 1.0, 1.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 1.0, 0.0, 0.0, 1.0, 1.0], is_bce=1)

        iteration = 5
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 1.0, 0.0, 1.0, 1.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 1.0, 0.0, 1.0, 1.0], is_bce=0)

        iteration = 6
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 1.0, 0.0, 1.0, 1.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 1.0, 0.0, 1.0, 1.0], is_bce=1)

        iteration = 7
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [1.0, 0.0, 0.0, 0.0, 10.0, 10.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [1.0, 0.0, 0.0, 0.0, 10.0, 10.0], is_bce=0)

        iteration = 8
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 10.0, 10.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 10.0, 10.0], is_bce=1)

        iteration = 9
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 1.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 1.0], is_bce=0)

        iteration = 10
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 1.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 1.0], is_bce=1)

        iteration = 11
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 0.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 0.0], is_bce=0)

        iteration = 12
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 0.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 1.0, 0.0], is_bce=1)

        iteration = 13
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=0)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=0)

        iteration = 14
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=1)

        iteration = 15
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.ALBEDO)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=1)
        self.iteration_table[str(iteration) + str(IterationTable.NetworkType.SHADING)] = IterationParameters(iteration, [10.0, 0.0, 0.0, 0.0, 0.0, 1.0], is_bce=1)


    def get_version(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_version()

    def get_l1_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(0)

    def get_lpip_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(1)

    def get_ssim_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(2)

    def get_gradient_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(3)

    def get_multiscale_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(4)

    def get_reflect_cons_weight(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].get_weight(5)

    def is_bce_enabled(self, iteration, network_type: NetworkType):
        return self.iteration_table[str(iteration) + str(network_type)].is_bce_enabled()

    def get_rgb_recon_weight(self):
        #return 1.0
        return 15.0

    def get_adv_weight(self):
        return 1.0

class IterationParameters():
    #6 weights total
    def __init__(self, iteration, weight_list, is_bce):
        self.iteration = iteration
        self.weight_list = weight_list
        self.is_bce = is_bce

    def get_version(self):
        return self.iteration

    def get_weight(self, index):
        if (index < len(self.weight_list)):
            return self.weight_list[index]
        else:
            # print("Weight index "+str(index)+ " not found. Returning 0.0")
            return 0.0

    def is_bce_enabled(self):
        return self.is_bce