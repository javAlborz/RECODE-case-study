import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = str(self.img_labels.iloc[idx, 0]) + '.jpg'
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


annotations_file = 'styles.csv'
img_dir = 'images_pp'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



from torchvision.models import resnet50, ResNet50_Weights

# Load the pre-trained ResNet model with specified weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Remove the last layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))
# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device


model.eval()  

features = []

with torch.no_grad():
    for inputs in train_loader:  
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.view(output.size(0), -1)  
        features.extend(output.cpu().numpy())



features_np = np.array(features)
# Compute similarity matrix
similarity_matrix = cosine_similarity(features_np, features_np)

import matplotlib.pyplot as plt


def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = tensor.clone().detach()  # Clone the tensor for safety
    tensor.mul_(std).add_(mean)  # Apply unnormalization
    tensor = tensor.clamp(0, 1)  # Clamp values to ensure they're within valid range
    return tensor

# Function to display an image
def imshow(tensor, ax=None):
    img = unnormalize(tensor)  # Unnormalize the image tensor
    img = img.permute(1, 2, 0).numpy()  # Convert to NumPy array in HxWxC format
    img = np.clip(img, 0, 1)  # Ensure values are within a valid range for display
    if ax is None:
        plt.imshow(img)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.axis('off')


test_index = 2
test_image_tensor = dataset[test_index]  
# Display the test image outside the subplots
plt.figure(figsize=(5, 5))
imshow(test_image_tensor)
plt.show()

# Find and display most similar images
similarity_scores = similarity_matrix[test_index]
most_similar_indices = similarity_scores.argsort()[-6:][::-1][1:]  # Exclude the first one (itself)

# Visualization of similar images
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, ax in enumerate(axes.flat):
    idx = most_similar_indices[i]
    similar_image_tensor = dataset[idx] 
    imshow(similar_image_tensor, ax=ax) 
plt.show()
