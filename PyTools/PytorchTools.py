import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def torch_device_config(index:int=0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(index))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

# Import and Export images to tensors
# the tensor size is [N, D, W, H], the range of value is within [0, 1]
def import_image_tensor(img_dir, size=None, device='cpu'):
    img = Image.open(img_dir)
    if size is not None:
        img.resize(size)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor

def export_image_tensor(img_tensor: torch.Tensor, img_dir):
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = np.ascontiguousarray(img)
    plt.imsave(img_dir, img)