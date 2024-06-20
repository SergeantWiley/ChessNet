import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
def estimate_remaining_time(start_time, current_iteration, total_iterations):
    elapsed_time = time.time() - start_time
    try:
        avg_time_per_iteration = elapsed_time / current_iteration
        remaining_iterations = total_iterations - current_iteration
        estimated_time_remaining = remaining_iterations * avg_time_per_iteration
        return elapsed_time, avg_time_per_iteration, estimated_time_remaining
    except:
        return None,None,None
def red_mask_intensity(image_path, output_dir):
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    mask = cv2.inRange(image_rgb, lower_red, upper_red)              
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    gray_mask_float = gray_mask.astype(np.float32)
    gray_mask_float /= 255.0
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, filename + 'masked.jpg')
    cv2.imwrite(output_path, (gray_mask_float * 255).astype(np.uint8))
    print(f"Grayscale intensity mask saved to {output_path}")

class NueralVal(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations.iloc[idx, 1:5].values.astype('float').reshape(-1, 4)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        target = {'boxes': boxes, 'labels': labels}
        return image, target

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = NueralVal(img_dir='MaskImages', 
                            annotations_file='annotations.csv', 
                            transform=transform)

train_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (serial number) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model setup
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (serial number) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 10
def train(train=True):
    if train:
        total_iterations = num_epochs * len(train_loader)
        progress = 0
        current_iteration = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for images, targets in train_loader:
                iter_start_time = time.time() #Log the start time
                #Training
                images = list(image.to(device) for image in images)  
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()
                current_iteration += 1

                #Progress Management
                progress = current_iteration / total_iterations
                iter_end_time = time.time()  # Log the end time
                time_passed = iter_end_time - iter_start_time
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / current_iteration
                time_left = round(((total_iterations - current_iteration) * avg_time_per_iteration) / 60, 2)
                
                print(f"Iteration {current_iteration}/{total_iterations} ({round(progress*100,2)}%), Current epoch Loss: {epoch_loss}, ETA: {time_left} min")

            print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')
            
        model_save_path = "fasterrcnn_model2.pth"
        torch.save(model.state_dict(), model_save_path)

train(False)
model_load_path = "fasterrcnn_model2.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode


def draw_bounding_boxes(image, predictions, threshold=0.65):
    # Convert the image from a tensor to a NumPy array and transpose the dimensions
    
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Loop over the predictions and draw the bounding boxes
    for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
        if score.item() >= threshold:
            x_min, y_min, x_max, y_max = box.float().cpu().numpy()
            label = predictions['labels'][i].item()

            # Convert coordinates to integers
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            print(x_min, y_min, x_max, y_max)

            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f'Score: {score:.2f}', color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Display the image with bounding boxes
    plt.axis('off')
    plt.show()
image_path = 'image17.jpg'
image_name = 'image17'
red_mask_intensity(image_path,"")
image = Image.open(f'{image_name}masked.jpg').convert("RGB")
image_tensor = F.to_tensor(image).to(device)
with torch.no_grad():
    predictions = model([image_tensor])[0]
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).to(device)
result_image = draw_bounding_boxes(image_tensor, predictions)


def run_inference_and_draw(image_path, model, device):
    red_mask_intensity(image_path,'')
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    #print(f"predictions: {predictions}")
    
    result_image = draw_bounding_boxes(image_tensor, predictions)
    
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()
