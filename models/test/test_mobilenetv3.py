import sys
import os
import torch
from models.mobilenetv3 import create_mobilenetv3
sys.path.append(os.path.abspath("./"))
model = create_mobilenetv3()
x=torch.randn(1,3,224,224)
y=model(x)
print(y)