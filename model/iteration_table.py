class IterationTable():

    def __init__(self):
        #initialize table
        self.iteration_table = {}

        iteration = 5
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, is_bce=0)

        iteration = 6
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, lpip_weight=1.0, ssim_weight=0.0, is_bce=1)

        iteration = 7
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, is_bce=0)

        iteration = 8
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=1.0, lpip_weight=10.0, ssim_weight=0.0, is_bce=1)

    def get_version(self, iteration):
        return self.iteration_table[iteration].get_version()

    def get_l1_weight(self, iteration):
        return self.iteration_table[iteration].get_l1_weight()

    def get_lpip_weight(self, iteration):
        return self.iteration_table[iteration].get_lpip_weight()

    def get_ssim_weight(self, iteration):
        return self.iteration_table[iteration].get_ssim_weight()

    def is_bce_enabled(self, iteration):
        return self.iteration_table[iteration].is_bce_enabled()

class IterationParameters():
    def __init__(self, iteration, l1_weight, lpip_weight, ssim_weight, is_bce):
        self.iteration = iteration
        self.l1_weight = l1_weight
        self.lpip_weight = lpip_weight
        self.ssim_weight = ssim_weight
        self.is_bce = is_bce

    def get_version(self):
        return self.iteration

    def get_l1_weight(self):
        return self.l1_weight

    def get_lpip_weight(self):
        return self.lpip_weight

    def get_ssim_weight(self):
        return self.ssim_weight

    def is_bce_enabled(self):
        return self.is_bce