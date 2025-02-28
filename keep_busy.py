import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to run a continuous GPU workload
def keep_gpu_busy():
    print("Starting GPU keep-alive loop...")
    while True:
        # Allocate a large tensor
        a = torch.randn(10000, 100000, device=device)
        b = torch.randn(100000, 10000, device=device)
        
        # Perform a matrix multiplication (intensive workload)
        c = torch.matmul(a, b)
        
        # Sync with GPU to ensure computation happens
        torch.cuda.synchronize()
        
        # Print log every 5 minutes
        print("GPU workload executed. Sleeping for 10 minutes...")
        time.sleep(300)  # Sleep for 5 minutes

if __name__ == "__main__":
    keep_gpu_busy()
