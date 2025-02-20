import sys
import os
import torch
sys.path.append(os.path.abspath("./"))

from models.mobilenetv3 import create_mobilenetv3
device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
print(device)
model = create_mobilenetv3(1).to(device)
x=torch.randn(1,3,224,224).to(device)
y=model(x)
print(y)