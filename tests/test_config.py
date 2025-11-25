# Default hyperparameters (simulating a training script)
depth = 12
learning_rate = 0.01
batch_size = 32
model_name = "baseline"

print("--- Initial Config ---")
print(f"depth={depth}, lr={learning_rate}, bs={batch_size}, name={model_name}")

# Load configurator
# This mimics how scripts load it: exec(open(...).read())
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

print("\n--- Running Configurator ---")
try:
    config_path = os.path.join(project_root, 'jujuchat', 'configurator.py')
    exec(open(config_path).read())
except Exception as e:
    print(f"❌ Configurator failed: {e}")

print("\n--- Final Config ---")
print(f"depth={depth}, lr={learning_rate}, bs={batch_size}, name={model_name}")

if depth == 20 and learning_rate == 0.0005:
    print("\n🎉 Phase 1 Configurator: PASSED")
else:
    print("\n⚠️  Configurator did not update values as expected (did you run with args?)")