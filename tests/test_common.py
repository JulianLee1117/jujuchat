import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jujuchat.common import compute_init, print0, get_base_dir, print_banner
import torch

# Test 1: Banner and Logging
print("\n--- Test 1: Banner ---")
print_banner()

# Test 2: Compute Init (Should detect MPS on Mac)
print("\n--- Test 2: Compute Init ---")
ddp, rank, local_rank, world_size, device = compute_init()
print0(f"✅ Device detected: {device}")
print0(f"✅ World size: {world_size}")
print0(f"✅ Rank: {rank}")

# Test 3: Base Directory
print("\n--- Test 3: Base Directory ---")
base_dir = get_base_dir()
print0(f"✅ Base directory: {base_dir}")

# Test 4: Tensor Operation on Device
print("\n--- Test 4: Tensor Operation ---")
x = torch.randn(5, 5, device=device)
y = x @ x.T
print0(f"✅ Matrix multiplication on {device}: {y.shape}")

print("\n🎉 Phase 1 Common Utilities: PASSED")