#!/usr/bin/env python

import torch

print(f"PyTorch CUDA: {torch.cuda.is_available()}\nCUDA Devices: {torch.cuda.device_count()}")
