'''Visualization of neural network activations and architecture.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ModelRunner(object):
    # Base class for model runners
    def __init__(self, model:nn.Module, num_epoch=3, learning_rate=1e-3, is_report=False):
        self.model = model
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.is_report = is_report
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses: list[float] = []

    def train(self, train_loader:torch.utils.data.DataLoader):
        """Train the model for a number of epochs"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epoch):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.is_report:
                    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.losses.append(avg_loss)
            
            if self.is_report:
                print(f"Epoch {epoch+1}/{self.num_epoch}, Loss: {avg_loss:.4f}")
        
        return self.losses


class ActivationExtractor:
    """Class to handle activation extraction using hooks"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.layer_names = []
    
    def get_activation(self, name):
        """Create a hook function that saves activations"""
        def hook(module, input, output):
            # Store both input and output
            self.activations[name] = {
                'input': input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone(),
                'output': output.detach().clone()
            }
        return hook
    
    def register_hooks(self, layers_to_monitor='all'):
        """Register forward hooks on specified layers"""
        
        if layers_to_monitor == 'all':
            # Monitor all Linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    hook = module.register_forward_hook(self.get_activation(name))
                    self.hooks.append(hook)
                    self.layer_names.append(name)
        else:
            # Monitor specific layers
            for layer_name in layers_to_monitor:
                module = dict(self.model.named_modules())[layer_name]
                hook = module.register_forward_hook(self.get_activation(layer_name))
                self.hooks.append(hook)
                self.layer_names.append(layer_name)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self):
        """Return stored activations"""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations = {}


class NetworkVisualizer(object):
    """Class to visualize network activations and architecture"""
    
    def __init__(self, model_runner):
        self.runner = runner
        self.extractor = ActivationExtractor(model)
    
def analyze_activations(activations, layer_names):
    """Analyze and visualize activation statistics"""
    
    print("=== Activation Analysis ===")
    
    for layer_name in layer_names:
        if layer_name in activations:
            output = activations[layer_name]['output']
            
            print(f"\nLayer: {layer_name}")
            print(f"  Shape: {output.shape}")
            print(f"  Mean: {output.mean().item():.4f}")
            print(f"  Std: {output.std().item():.4f}")
            print(f"  Min: {output.min().item():.4f}")
            print(f"  Max: {output.max().item():.4f}")
            
            # Check for dead neurons (always zero output)
            if len(output.shape) > 1:
                dead_neurons = (output.abs().sum(dim=0) == 0).sum().item()
                total_neurons = output.shape[1]
                print(f"  Dead neurons: {dead_neurons}/{total_neurons}")

def visualize_activations(activations, layer_names, sample_idx=0):
    """Visualize activations for a specific sample"""
    
    fig, axes = plt.subplots(2, len(layer_names), figsize=(4*len(layer_names), 8))
    if len(layer_names) == 1:
        axes = axes.reshape(2, 1)
    
    for i, layer_name in enumerate(layer_names):
        if layer_name in activations:
            output = activations[layer_name]['output']
            
            # Plot activation values
            axes[0, i].bar(range(output.shape[1]), output[sample_idx].detach().numpy())
            axes[0, i].set_title(f'{layer_name}\nActivation Values')
            axes[0, i].set_xlabel('Neuron Index')
            axes[0, i].set_ylabel('Activation')
            
            # Plot activation histogram
            axes[1, i].hist(output[sample_idx].detach().numpy(), bins=20, alpha=0.7)
            axes[1, i].set_title(f'{layer_name}\nActivation Distribution')
            axes[1, i].set_xlabel('Activation Value')
            axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def gradient_hook_example(model):
    """Example of using backward hooks to capture gradients"""
    
    gradients = {}
    
    def get_gradient_hook(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = {
                'grad_input': grad_input[0].detach().clone() if grad_input[0] is not None else None,
                'grad_output': grad_output[0].detach().clone() if grad_output[0] is not None else None
            }
        return hook
    
    # Register backward hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_full_backward_hook(get_gradient_hook(name))
            hooks.append(hook)
    
    return gradients, hooks

# Example usage
if __name__ == "__main__":
    # Create model and data
    model = LinearNetwork(input_size=8, hidden_sizes=[16, 12, 8], output_size=1)
    
    # Create sample data
    batch_size = 32
    input_data = torch.randn(batch_size, 8)
    targets = [[1.] if np.mean(v) < -0.2 else [2.] if np.mean(v) < 0.2 else [3.] for v in input_data.numpy().tolist()]
    target = torch.tensor(targets, dtype=torch.float)
    #target = torch.randint(0, 3, (batch_size,))
    
    print("Model architecture:")
    print(model)
    
    # === FORWARD HOOKS EXAMPLE ===
    print("\n" + "="*50)
    print("FORWARD HOOKS EXAMPLE")
    print("="*50)
    
    # Create activation extractor
    extractor = ActivationExtractor(model)
    
    # Register hooks for all Linear layers
    extractor.register_hooks(layers_to_monitor='all')
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    
    # Get activations
    activations = extractor.get_activations()
    
    # Analyze activations
    analyze_activations(activations, extractor.layer_names)
    
    # Visualize activations
    visualize_activations(activations, extractor.layer_names, sample_idx=0)
    
    # Clean up
    extractor.remove_hooks()
    
    # === GRADIENT HOOKS EXAMPLE ===
    print("\n" + "="*50)
    print("GRADIENT HOOKS EXAMPLE")
    print("="*50)
    
    # Set up gradient hooks
    gradients, grad_hooks = gradient_hook_example(model)
    
    # Forward and backward pass
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    output = model(input_data)
    import pdb; pdb.set_trace()  # Debugging point to inspect output
    loss = criterion(output, target)
    loss.backward()
    
    # Analyze gradients
    print("\n=== Gradient Analysis ===")
    for layer_name, grad_data in gradients.items():
        if grad_data['grad_output'] is not None:
            grad = grad_data['grad_output']
            print(f"\nLayer: {layer_name}")
            print(f"  Gradient shape: {grad.shape}")
            print(f"  Gradient norm: {grad.norm().item():.6f}")
            print(f"  Gradient mean: {grad.mean().item():.6f}")
    
    # Clean up gradient hooks
    for hook in grad_hooks:
        hook.remove()
    
    # === SPECIFIC LAYER MONITORING ===
    print("\n" + "="*50)
    print("SPECIFIC LAYER MONITORING")
    print("="*50)
    
    # Monitor only specific layers
    extractor2 = ActivationExtractor(model)
    specific_layers = ['network.0', 'network.2']  # First and second Linear layers
    extractor2.register_hooks(layers_to_monitor=specific_layers)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_data)
    
    # Get and analyze specific activations
    specific_activations = extractor2.get_activations()
    analyze_activations(specific_activations, specific_layers)
    
    # Clean up
    extractor2.remove_hooks()
    
    print(f"\n=== Hook Usage Complete ===")
    print("Hooks allow you to:")
    print("1. Extract intermediate activations without modifying the model")
    print("2. Monitor gradients during backpropagation") 
    print("3. Debug network behavior and identify issues")
    print("4. Analyze feature representations at different layers")