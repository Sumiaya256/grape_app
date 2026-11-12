import os, torch

SRC = "weights"            # your original big files
DST = "weights_small"      # smaller copies go here
os.makedirs(DST, exist_ok=True)

for f in os.listdir(SRC):
    if f.endswith(".pth"):
        sd = torch.load(os.path.join(SRC, f), map_location="cpu")
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        torch.save(sd, os.path.join(DST, f))
        print(f, round(os.path.getsize(os.path.join(SRC, f)) / 1e6, 1), "MB ->",
              round(os.path.getsize(os.path.join(DST, f)) / 1e6, 1), "MB")