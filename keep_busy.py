import torch
import time

# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create two large tensors
size = 1024  # Adjust size based on your GPU memory
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

# Run one matmul every 60 seconds
try:
    while True:
        start = time.time()
        print(f"Running matmul at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        print("Done. Sleeping for 30 seconds...\n")
        # time.sleep(max(0, 60 - (time.time() - start)))
except KeyboardInterrupt:
    print("Stopped by user.")
