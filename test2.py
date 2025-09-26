import torch
import sys

# Get the file path from the command line argument
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("Usage: python inspect_tensor.py <path_to_your_pt_file>")
    sys.exit(1)

print(f"Inspecting file: {file_path}")

try:
    # Load the tensor from the file
    tensor = torch.load(file_path)

    # Check if it's a tensor
    if isinstance(tensor, torch.Tensor):
        print("File loaded successfully. It contains a PyTorch Tensor.")
        print(f"  - Data type: {tensor.dtype}")
        print(f"  - Shape: {tensor.shape}")

        # If the tensor is small, print its content
        if tensor.numel() > 0 and tensor.numel() < 20:
            print(f"  - Content: {tensor}")
        elif tensor.numel() == 0:
            print("\n!!! WARNING: The tensor is EMPTY. This is likely the cause of your error. !!!")
        else:
            print(f"  - First 5 elements: {tensor[:5]}")

    else:
        # The file might contain a dictionary or other object
        print(f"File loaded, but it is not a tensor. It is a {type(tensor)}")
        if isinstance(tensor, dict):
            print("  - Keys:", tensor.keys())


except Exception as e:
    print(f"\n--- An error occurred while trying to load the file ---")
    print(e)
    print("This could mean the file is corrupted, not a valid PyTorch file, or was saved incorrectly.")