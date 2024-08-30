import PyTools.PytorchTools as PT

if __name__ == "__main__":
    device = PT.torch_device_config()
    img_tensor = PT.import_image_tensor(img_dir="./TestSets/paw.png", device=device)
    print(img_tensor.size())
    #PT.export_image_tensor(img_tensor, "./TestSets/test.png")