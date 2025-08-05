'''Visualizes neurons during training.'''

"""
Creates heat maps that visualize neuro activations during training.
The horizontal axis is the layer; the vertical access is the neural.
"""

import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms    

ActivationMap = collections.namedtuple('ActivationMap', ['epoch', 'layer_name', 'activation_arr'])

class TrainingVisualizer(object):

    def __init__(self, model:nn.Module):
        self.model = model
        self.activation_maps = []

    def addActivationMap(self, epoch:int):
        """Adds activations"""
        def hook(module, input, output):
            self.activations.append(output.detach())
        #
        layer_names = [name for name, _ in self.model.named_parameters()]
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        layer.register_forward_hook(hook)
    
    def visualize(self):
        """_summary_

        Args:
            model (_type_): _description_
            train_loader (_type_): _description_
            test_loader (_type_): _description_
            epochs (int, optional): _description_. Defaults to 10.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
            
            if self.mode == 'test':
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                print(f'Test Accuracy: {accuracy:.4f}')