import os
import subprocess
import sys

import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available.")

check_cuda()

