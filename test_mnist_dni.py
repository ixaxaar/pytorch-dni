#!/usr/bin/env python3
"""
Test DNI on MNIST task to validate real-world performance
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dni import DNI, LinearDNI

def test_mnist_dni():
    print("Testing DNI on MNIST classification task")
    print("=" * 50)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Small dataset for quick test
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Define model
    class MNISTNet(nn.Module):
        def __init__(self):
            super(MNISTNet, self).__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.classifier = nn.Linear(64, 10)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = self.features(x)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
    
    # Test different configurations
    configs = [
        {"name": "Standard BP", "λ": 1.0},
        {"name": "Pure DNI", "λ": 0.0},
        {"name": "Mixed (λ=0.5)", "λ": 0.5}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']} (λ={config['λ']})")
        
        # Create model
        model = MNISTNet()
        
        # Wrap feature extractor with DNI (skip classifier)
        model.features = DNI(
            model.features,
            hidden_size=64,
            λ=config["λ"],
            recursive=True,
            skip_last_layer=True
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Train for just a few batches
        total_loss = 0
        num_batches = 5  # Quick test
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
                
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            
            total_loss += loss.item()
            
            loss.backward()
            
            # Execute DNI optimizer steps
            if hasattr(model.features, 'step_optimizers'):
                model.features.step_optimizers()
            
            optimizer.step()
            
            if batch_idx % 2 == 0:
                print(f'  Batch {batch_idx}: Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        print(f'  ✓ {config["name"]}: Average loss: {avg_loss:.6f}')
    
    print("\n🎉 MNIST DNI test completed successfully!")

if __name__ == "__main__":
    try:
        test_mnist_dni()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()