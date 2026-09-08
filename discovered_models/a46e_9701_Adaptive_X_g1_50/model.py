"""
Auto-Invented Neural Architecture: Adaptive_X_g1_50
ID: a46e_9701
Generation: 1
Parent: a46e565a
Task Domain: sequence
Mutated from Adaptive_Engine_v0_235 (Gen 0) to explore topological variations and higher efficiency.
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

class Adaptive_X_g1_50(nn.Module):
    """
    Adaptive_X_g1_50 PyTorch Implementation.
    """
    def __init__(self, input_dim: int = 16, output_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64

        # Input embedding projection
        self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # Discovered Architecture Layers
        self.layer_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.layer_1 = LiquidMemoryGate(dim=64)
        self.layer_2 = MultiScaleWaveletConv1D(channels=64)
        self.layer_3 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.layer_4 = MultiScaleWaveletConv1D(channels=64)

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
        # Execute Layer 0 (AttentionBlock)
        attn_out, _ = self.layer_0(h, h, h)
        h = h + attn_out
        saved_states[0] = h
        # Execute Layer 1 (LiquidMemoryGate)
        h = self.layer_1(h)
        saved_states[1] = h
        # Execute Layer 2 (MultiScaleWaveletConv1D)
        h = h + 0.5 * (saved_states[1])
        h = self.layer_2(h)
        saved_states[2] = h
        # Execute Layer 3 (AttentionBlock)
        attn_out, _ = self.layer_3(h, h, h)
        h = h + attn_out
        saved_states[3] = h
        # Execute Layer 4 (MultiScaleWaveletConv1D)
        h = h + 0.5 * (saved_states[1])
        h = self.layer_4(h)
        saved_states[4] = h

        # Pool sequence if needed and apply head
        if is_seq and h.dim() == 3:
            h_pooled = h.mean(dim=1)  # Global average pooling over sequence
        else:
            h_pooled = h

        out = self.out_head(h_pooled)
        return out

# Standalone test execution block
if __name__ == "__main__":
    print("Testing Adaptive_X_g1_50 on target hardware...")
    from ai_builder.backend.hardware import get_hardware_engine
    engine = get_hardware_engine()
    device = engine.get_device()
    print(f"Device: {device} ({engine.backend_name})")

    model = Adaptive_X_g1_50().to(device)
    test_input = torch.randn(4, 10, 16).to(device)
    output = model(test_input)
    print(f"Input shape: {test_input.shape} -> Output shape: {output.shape}")
    print("Model test successful!")