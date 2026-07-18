import torch
import sys

checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/gt_but_full/checkpoints/agent_80000.pt"
output_path = checkpoint_path.replace(".pt", "_orientation_only.pt")

checkpoint = torch.load(checkpoint_path, map_location="cpu")

# ✅ ПРАВИЛЬНО: ищем внутри critic_1
orientation_weights = {}

if "critic_1" in checkpoint and isinstance(checkpoint["critic_1"], dict):
    for key, value in checkpoint["critic_1"].items():
        if "orientation_module" in key:
            # Убрать префикс "orientation_module."
            new_key = key.replace("orientation_module.", "")
            orientation_weights[new_key] = value

if not orientation_weights:
    print(f"❌ No orientation_module weights found")
    sys.exit(1)

torch.save(orientation_weights, output_path)
print(f"✅ Saved {len(orientation_weights)} weights to {output_path}")
print(f"   Keys: {list(orientation_weights.keys())}")
