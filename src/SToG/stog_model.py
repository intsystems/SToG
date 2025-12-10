import torch
import torch.nn as nn
from typing import List, Tuple
from .base import BaseFeatureSelector


class SToGModel(nn.Module):
    def __init__(self, model: nn.Sequential, selection_layers: List[Tuple[int, BaseFeatureSelector]]):
        '''
            Adding SToG functionality to nn.Sequential model through SToG feature selection layers.
        

            Args:
                model: model of type nn.Sequential (without feature selection layers)
                selection_layers: List of tuples of layer number and feature selection layer object
                
            Returns:
                PyTorch sequential model with added feature selection layers
        '''

        super().__init__()

        for el in selection_layers:
            if el[0] < 0 or el[0] > len(list(model)) - 1:
                raise ValueError("Selection layers must be in the range [1, num_operations]")

        layers = list(model)

        for idx, layer in sorted(selection_layers, key=lambda x: x[0], reverse=True):
            layers.insert(idx, layer)

        self.sel_layer_indices = [el[0] + i for i, el in enumerate(selection_layers)]

        self.features_to_select = sum([el[1].input_dim for el in selection_layers])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def regularization_loss(self):
        return sum(self.layers[i].regularization_loss() for i in self.sel_layer_indices)