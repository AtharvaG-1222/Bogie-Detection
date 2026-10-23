import torch

ckpt_path = "parameters/params.pth"  # or whichever file showed Epoch 70
dino1_path = "parameters/vit_bogie_dino1.pth"

ckpt = torch.load(ckpt_path, map_location="cpu")
dino1 = torch.load(dino1_path, map_location="cpu")

# Extract backbone weights from the checkpoint
ckpt_weights = ckpt.get("backbone_state_dict", ckpt.get("student_state_dict", ckpt))

# Compare a deep attention layer
key = "blocks.11.attn.qkv.weight"
w_ckpt = ckpt_weights.get(key, ckpt_weights.get(f"backbone.{key}"))
w_dino1 = dino1.get(key, dino1.get(f"backbone.{key}"))

diff = torch.norm(w_ckpt - w_dino1).item()
print(f"Weight difference norm: {diff:.6f}")

if diff < 1e-4:
    print("[+] vit_bogie_dino1.pth IS the final exported model from the Epoch 70 training run!")
else:
    print("[-] vit_bogie_dino1.pth has different weights (this is the 10-image trial run).")
    print(f"    Use the Epoch 70 file ({ckpt_path}) as your trained model.")