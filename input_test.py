import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset import DataModule

# Define the transform for ResNet (expects 3-channel input)
import torchvision.transforms as transforms

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform_for_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Path to your CSV file
CSV_PATH = 'dataset/icml_face_data.csv'
MODEL_NAME = 'resnet_fer'

# Create the DataModule
dm = DataModule(
    path=CSV_PATH,
    model_name=MODEL_NAME,
    transform=transform_for_resnet,
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    shuffle=True
)

# Get the validation loader
val_loader = dm.get_val_loader()

# Get one batch
label, image_tensor = next(iter(val_loader))
image_tensor = image_tensor.squeeze(0)  # Remove batch dimension

print(f"Tensor shape: {image_tensor.shape}")
print(f"Min value (normalized): {image_tensor.min():.4f}")
print(f"Max value (normalized): {image_tensor.max():.4f}")

# Unnormalize and visualize
def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

unnormalized_image = unnormalize(image_tensor, imagenet_mean, imagenet_std)
image_to_show = unnormalized_image.permute(1, 2, 0).numpy()
image_to_show = np.clip(image_to_show, 0, 1)

import matplotlib.pyplot as plt
plt.imshow(image_to_show)
plt.title(f"Label: {label.item()}")
plt.axis('off')
plt.show()