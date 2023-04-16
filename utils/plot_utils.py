# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:02:01 2020

@author: delgallegon
"""
from matplotlib.lines import Line2D

import global_config
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import visdom

SALIKSIK_SERVER = "192.168.134.223" #IMPORTmsANT: No HTTP

class VisdomReporter:
    _sharedInstance = None

    @staticmethod
    def initialize():
        VisdomReporter._sharedInstance = VisdomReporter()

    @staticmethod
    def getInstance():
        return VisdomReporter._sharedInstance

    def __init__(self):
        if(global_config.server_config == 0):
            self.vis = visdom.Visdom(SALIKSIK_SERVER, use_incoming_socket=False, port=8097) #TODO: Note that this is set to TRUE for observation.
        elif(global_config.server_config == 1):
            self.vis = None
        elif(global_config.plot_enabled == 0):
            self.vis = None
        else:
            self.vis= visdom.Visdom()
        
        self.image_windows = {}
        self.loss_windows = {}
        self.text_windows = {}
    
    def plot_image(self, img_tensor, caption, normalize = True):
        if(global_config.plot_enabled == 0):
            return

        img_group = vutils.make_grid(img_tensor[:16], nrow = 8, padding=2, normalize=normalize).cpu()
        if hash(caption) not in self.image_windows:
            self.image_windows[hash(caption)] = self.vis.images(img_group, opts = dict(caption = caption))
        else:
            self.vis.images(img_group, win = self.image_windows[hash(caption)], opts = dict(caption = caption))

    def plot_text(self, text):
        if(global_config.plot_enabled == 0):
            return

        if(hash(text) not in self.text_windows):
            self.text_windows[hash(text)] = self.vis.text(text, opts = dict(caption = text))
        else:
            self.vis.text(text, win = self.text_windows[hash(text)], opts = dict(caption = text))

    def plot_grad_flow(self, named_parameters, caption):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        if hash(caption) not in self.loss_windows:
            self.loss_windows[hash(caption)] = self.vis.matplot(plt, opts = dict(caption = caption))
        else:
            self.vis.matplot(plt, win = self.loss_windows[hash(caption)], opts = dict(caption = caption))

    def plot_finegrain_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        if(global_config.plot_enabled == 0):
            return
        
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
        index = 0
        
        x = [i for i in range(iteration, iteration + len(losses_dict["g_loss"]))]
        COLS = 3; ROWS = 4
        fig, ax = plt.subplots(ROWS, COLS, sharex=True)
        fig.set_size_inches(9, 9)
        fig.tight_layout()

        row = 0
        col = 0
        for i in range(0, len(loss_keys)):
            if(i == 1):
                ax[row, col].plot(x, losses_dict[loss_keys[i]], color=colors[i], label=loss_key + " " + str(caption_dict[caption_keys[i]]))
                col = col + 1
            elif(np.mean(losses_dict[loss_keys[i]]) > 0.0): #only display those > 0.0
                ax[row, col].plot(x, losses_dict[loss_keys[i]], color=colors[i], label=str(caption_dict[caption_keys[i]]))
                col = col + 1

            if(col == COLS):
                row = row + 1
                col = 0

        # for i in range(ROWS):
        #     for j in range(COLS):
        #         if(index < len(loss_keys)):
        #             if(index == 1):
        #                 ax[i, j].plot(x, losses_dict[loss_keys[index]], color=colors[index], label= loss_key + " " +str(caption_dict[caption_keys[index]]))
        #             elif (np.mean(losses_dict[loss_keys[index]]) > 0.0): #only display those > 0.0
        #                 ax[i, j].plot(x, losses_dict[loss_keys[index]], color = colors[index], label = str(caption_dict[caption_keys[index]]))
        #             index = index + 1
        #         else:
        #             break
    
        fig.legend(loc = 'lower right')
        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(label)))
        else:
            self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(label)))
          
        plt.show()

    # def plot_train_test_loss(self, loss_key, iteration, train_losses, test_losses, train_caption, test_caption):
    #     colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
    #
    #     x1 = [i for i in range(iteration, iteration + len(train_losses))]
    #     x2 = [i for i in range(iteration, iteration + len(test_losses))]
    #
    #     plt.plot(x1, train_losses, color=colors[0], label=str(train_caption))
    #     plt.plot(x2, test_losses, color=colors[1], label=str(test_caption))
    #     plt.legend(loc='lower right')
    #
    #     if loss_key not in self.loss_windows:
    #         self.loss_windows[loss_key] = self.vis.matplot(plt, opts=dict(caption="Losses" + " " + str(global_config)))
    #     else:
    #         self.vis.matplot(plt, win=self.loss_windows[loss_key], opts=dict(caption="Losses" + " " + str(global_config)))
    #
    #     plt.show()

    def plot_train_test_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        if (global_config.plot_enabled == 0):
            return
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']

        x = [i for i in range(iteration, iteration + len(losses_dict["TRAIN_LOSS_KEY"]))]
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())

        plt.plot(x, losses_dict[loss_keys[0]], colors[0], label=str(caption_dict[caption_keys[0]]))
        plt.plot(x, losses_dict[loss_keys[1]], colors[1], label=str(caption_dict[caption_keys[1]]))
        plt.legend(loc='lower right')

        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(label)))
        else:
            self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(label)))