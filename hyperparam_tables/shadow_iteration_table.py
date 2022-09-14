from enum import Enum

class ShadowIterationTable():
    def __init__(self):
        #initialize table
        self.iteration_table = {}

        iteration = 11
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [10.0, 1.0, 0.0], is_bce=0)

        iteration = 12
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [10.0, 1.0, 0.0], is_bce=1)

        iteration = 13
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [1.0, 10.0, 0.0], is_bce=0)

        iteration = 14
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [1.0, 10.0, 0.0], is_bce=1)

        iteration = 15
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [1.0, 1.0, 0.0], is_bce=0)

        iteration = 16
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [1.0, 1.0, 0.0], is_bce=1)

        iteration = 17
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [10.0, 1.0, 1.0], is_bce=0)

        iteration = 18
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [10.0, 1.0, 1.0], is_bce=1)

        iteration = 19
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [100.0, 10.0, 0.0], is_bce=0)

        iteration = 20
        self.iteration_table[str(iteration)] = IterationParameters(iteration, [100.0, 10.0, 0.0], is_bce=1)

    def get_version(self, iteration):
        return self.iteration_table[str(iteration)].get_version()

    def get_l1_weight(self, iteration):
        return self.iteration_table[str(iteration)].get_weight(0)

    def get_lpip_weight(self, iteration):
        return self.iteration_table[str(iteration)].get_weight(1)

    def get_masking_weight(self, iteration):
        return self.iteration_table[str(iteration)].get_weight(2)

    def is_bce_enabled(self, iteration):
        return self.iteration_table[str(iteration)].is_bce_enabled()

    def get_gammabeta_weight(self, iteration):
        if(iteration < 10):
            return 100.0
        else:
            return 10.0

    def get_rgb_recon_weight(self):
        return 20.0

    def get_rgb_lpips_weight(self):
        return 1.0

    def get_rgb_adv_weight(self):
        return 1.0

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