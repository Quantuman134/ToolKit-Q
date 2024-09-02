import PyTools.PytorchTools as PT
import torch

if __name__ == "__main__":
    device = PT.torch_device_config()
    img_tensor = PT.import_image_tensor(img_dir="./TestSets/paw.png", device=device)
    print(img_tensor.size())
    img_tensor = PT.import_image_tensor(img_dir="./TestSets/paw.png", device=device, size=(512, 512), dtype=torch.float64)
    print(img_tensor.size())
    print(img_tensor.dtype)
    #PT.export_image_tensor(img_tensor, "./TestSets/test.png")