#!/usr/bin/env python3
# Test RNN with DNI
import sys
import os
sys.path.insert(0, '.')

try:
    import torch
    import torch.nn as nn
    from dni import DNI, RNNDNI
    
    print("Testing RNN with DNI...")
    
    # Create simple RNN
    rnn = nn.LSTM(input_size=20, hidden_size=30, num_layers=1, batch_first=True)
    
    # Wrap with DNI - simpler configuration
    dni_rnn = DNI(
        rnn,
        dni_network=RNNDNI,
        hidden_size=16,
        λ=0.5,
        recursive=False,
        format_fn=lambda x: x[0]  # Extract only the output tensor from RNN
    )
    
    print("✓ DNI RNN created successfully")
    
    # Test data
    batch_size = 5
    seq_len = 10
    input_size = 20
    
    input_data = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    
    # Enable training mode
    dni_rnn.train()
    
    # Forward pass
    output, hidden = dni_rnn(input_data)
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Simple loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    # Execute DNI optimizer steps
    dni_rnn.step_optimizers()
    print("✓ DNI optimizer steps executed")
    
    print("\n🎉 RNN DNI test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()