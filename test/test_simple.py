#!/usr/bin/env python3
# Simple test to check if DNI works
import sys
import os
sys.path.insert(0, '.')

try:
    # Basic imports test
    print("Testing imports...")
    import torch
    import torch.nn as nn
    from dni import DNI, LinearDNI
    print("✓ All imports successful")
    
    # Create a simple test network
    print("Creating test network...")
    net = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Wrap with DNI
    print("Wrapping with DNI...")
    dni_net = DNI(
        net,
        hidden_size=16,
        λ=0.5,
        recursive=True
    )
    print("✓ DNI network created successfully")
    
    # Test forward pass
    print("Testing forward pass...")
    input_data = torch.randn(5, 10, requires_grad=True)
    
    # Enable training mode
    dni_net.train()
    
    output = dni_net(input_data)
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Test backward pass
    print("Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n🎉 All tests passed! DNI implementation appears to be working.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("PyTorch is required but not available in the environment")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()