# Веса сети символов → npz для symclass.SymNet: venv312\Scripts\python.exe export_sym.py best.pt <куда>
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("train_sym", ROOT / "train_sym.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
net = T.Net(len(T.CLASSES))
net.load_state_dict(torch.load(ROOT / sys.argv[1], map_location="cpu"))
net.eval()
out = {"classes": np.array(T.CLASSES), "fc1w": net.fc1.weight.detach().numpy(), "fc1b": net.fc1.bias.detach().numpy(),
       "fc2w": net.fc2.weight.detach().numpy(), "fc2b": net.fc2.bias.detach().numpy()}
i = 0
for m in net.body:
    if isinstance(m, torch.nn.Conv2d):
        out[f"k{i}"], out[f"w{i}"] = np.array("conv"), m.weight.detach().numpy().astype(np.float32)
    elif isinstance(m, torch.nn.BatchNorm2d):
        out[f"k{i}"] = np.array("norm")
        for a, v in (("mean", m.running_mean), ("var", m.running_var), ("gamma", m.weight.detach()), ("beta", m.bias.detach())):
            out[f"{a}{i}"] = v.numpy()
    elif isinstance(m, torch.nn.ReLU):
        out[f"k{i}"] = np.array("relu")
    else:
        out[f"k{i}"] = np.array("pool")
    i += 1
out["n"] = np.array(i)
np.savez_compressed(sys.argv[2], **out)
print("файл", round(Path(sys.argv[2]).stat().st_size / 1e6, 2), "МБ")
