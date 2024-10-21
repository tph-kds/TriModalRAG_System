import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print(f"Using device: {device}")

if __name__ == "__main__":
    assert torch.device("cuda" if torch.cuda.is_available() else "cpu") == "gpu"
    

