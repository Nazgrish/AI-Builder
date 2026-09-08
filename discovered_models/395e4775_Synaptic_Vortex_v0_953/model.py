"""
Auto-Invented Neural Architecture: Synaptic_Vortex_v0_953
ID: 395e4775
Generation: 0
Parent: genesis
Task Domain: sequence
Hypothesis: Combining MultiScaleWaveletConv1D, LinearBlock with AdaptiveHarmonicActivation and 0 dynamic skip connections will improve gradient flow, enhance representation capacity, and maximize throughput on AMD Radeon hardware.
Target Hardware: AMD Radeon (DirectML / ROCm / OpenCL)
Synthesized by AI Builder Autonomous AI Engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_builder.inventor.primitives import (
    AdaptiveHarmonicActivation,
    DynamicGatedSwish,
    FractalRoutingBlock,
    LiquidMemoryGate,
    MultiScaleWaveletConv1D
)

class Synaptic_Vortex_v0_953(nn.Module):
    """
    Synaptic_Vortex_v0_953 PyTorch Implementation.
    """
    def __init__(self, input_dim: int = 16, output_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 32

        # Input embedding projection
        self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # Discovered Architecture Layers
        self.layer_0 = MultiScaleWaveletConv1D(channels=32)
        self.layer_1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(32)
        )

        # Output projection head
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            AdaptiveHarmonicActivation(),
            nn.Linear(self.hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Features] or [Batch, SeqLen, Features]
        is_seq = (x.dim() == 3)
        
        h = self.in_proj(x)
        h = self.in_norm(h)

        saved_states = {}
        # Execute Layer 0 (MultiScaleWaveletConv1D)
        h = self.layer_0(h)
        saved_states[0] = h
        # Execute Layer 1 (LinearBlock)
        h = self.layer_1(h)
        saved_states[1] = h

        # Pool sequence if needed and apply head
        if is_seq and h.dim() == 3:
            h_pooled = h.mean(dim=1)  # Global average pooling over sequence
        else:
            h_pooled = h

        out = self.out_head(h_pooled)
        return out

# Standalone test execution block
if __name__ == "__main__":
    print("Testing Synaptic_Vortex_v0_953 on target hardware...")
    from ai_builder.backend.hardware import get_hardware_engine
    engine = get_hardware_engine()
    device = engine.get_device()
    print(f"Device: {device} ({engine.backend_name})")

    model = Synaptic_Vortex_v0_953().to(device)
    test_input = torch.randn(4, 10, 16).to(device)
    output = model(test_input)
    print(f"Input shape: {test_input.shape} -> Output shape: {output.shape}")
    print("Model test successful!")