import argparse
import math
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "gb",
    type=float,
    help="Amount of GPU memory to allocate (GiB)",
)
args = parser.parse_args()

elements = math.ceil((args.gb * 1024**3) / 2)  # fp16 = 2 bytes/element
x = torch.zeros(elements, dtype=torch.float16, device="cuda")

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"Allocated: {allocated:.2f} GiB")
print(f"Reserved:  {reserved:.2f} GiB")
print("GPU memory allocated. Sleeping forever...")

while True:
    time.sleep(3600)