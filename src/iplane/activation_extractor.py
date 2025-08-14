'''Extracts activations from a pytorch model.'''
# author: Claude

import iplane.constants as cn # type: ignore

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class ActivationExtractor:
    """Simple hook-based activation extractor"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names=None):
        """Register hooks to extract activations"""
        
        def hook_fn(name):
            def hook(module, input, output):
                # Store activation (detach to save memory)
                self.activations[name] = output.detach()
            return hook
        
        # Hook all layers if none specified
        if layer_names is None:
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Only leaf modules
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)
        else:
            # Hook only specified layers
            for name, module in self.model.named_modules():
                if name in layer_names:
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)
    
    def get_activations(self, x):
        """Run forward pass and return activations"""
        self.activations = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.activations
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []