import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = ('cpu')
print("Using Device:", device)

class MaskImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("L")  # Convert image to grayscale
        bbox = self.annotations.iloc[idx, 1:5].values
        bbox = bbox.astype('float').reshape(-1, 4)
        
        if self.transform:
            image = self.transform(image)
        
        return image.float(), torch.tensor(bbox).float()

# Define the transformations
transform = transforms.Compose([
    #transforms.Resize(244,244),
    transforms.ToTensor(),
])

# Create the dataset
dataset = MaskImagesDataset(csv_file='annotations.csv', root_dir='MaskImages', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print("Dataset Created")
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 960 * 200, 500)
        self.fc2 = nn.Linear(500, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 960 * 200)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleCNN().to(device)
print("Model created")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
print("Starting Training")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (images, bboxes) in enumerate(dataloader):
        images = images.to(device)
        bboxes = bboxes.to(device)
        print("Dataset Loaded")
        outputs = model(images)
        print("Outputs Calculated")
        loss = criterion(outputs, bboxes.view(-1, 4))
        print("Loss Calculated")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for images, bboxes in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, bboxes.view(-1, 4))
            total_loss += loss.item()
    print(f'Average Loss: {total_loss / len(dataloader):.4f}')

evaluate_model(model, dataloader)

