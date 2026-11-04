# Из весов PyTorch в npz для ocr.NetReader; проверка совпадения с PyTorch на настоящих надписях.
#   venv312\Scripts\python.exe export.py best.pt <куда>
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("train_ocr", ROOT / "train_ocr.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
net = T.Net(len(T.ALPHABET) + 1)
net.load_state_dict(torch.load(ROOT / sys.argv[1], map_location="cpu"))
net.eval()
out = {"alphabet": np.array(T.ALPHABET)}
i = 0


def add(kind, **kw):
    global i
    out[f"k{i}"] = np.array(kind)
    for k, v in kw.items():
        out[f"{k}{i}"] = v
    i += 1


for m in net.body:
    if isinstance(m, torch.nn.Conv2d):
        add("conv", w=m.weight.detach().numpy().astype(np.float32), p=np.array(m.padding))
    elif isinstance(m, torch.nn.BatchNorm2d):
        add("norm", mean=m.running_mean.numpy(), var=m.running_var.numpy(), gamma=m.weight.detach().numpy(), beta=m.bias.detach().numpy())
    elif isinstance(m, torch.nn.ReLU):
        add("relu")
    elif isinstance(m, torch.nn.MaxPool2d):
        k = m.kernel_size
        add("pool", p=np.array(2 if k == 2 else 1))
head = [m for m in net.head if isinstance(m, torch.nn.Conv1d)]
add("conv", w=head[0].weight.detach().numpy()[:, :, None, :].astype(np.float32), p=np.array((0, 2)))
add("relu")
add("conv", w=head[1].weight.detach().numpy()[:, :, None, :].astype(np.float32), p=np.array((0, 0)))
out["n"] = np.array(i)
np.savez_compressed(sys.argv[2], **out)
print("файл", round(Path(sys.argv[2]).stat().st_size / 1e6, 2), "МБ, слоёв", i)
