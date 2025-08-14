# german_signs/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import imageio.v2 as imageio
import os
import argparse
import numpy as np
from tqdm import tqdm

# Import your custom augmentation classes
from camera_occlusion.camera_noise import Rain, Dust

# ====================================================================================
#  1. CUSTOM DATASET FOR GTSRB WITH ON-THE-FLY NOISE
# ====================================================================================

class GTSRBDataset(Dataset):
    """
    Custom PyTorch Dataset for the GTSRB dataset.
    It finds all images, assigns labels based on the folder structure,
    and applies a specified noise effect on-the-fly.
    """
    def __init__(self, root_dir, transform=None, effect=None):
        """
        Args:
            root_dir (string): Directory with all the class folders.
            transform (callable, optional): Optional standard transform to be applied on a sample.
            effect (Effect, optional): An instance of Rain, Dust, or another custom effect.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.effect = effect
        self.samples = []
        self.num_classes = 0

        # Discover classes and image paths
        # Assumes root_dir contains subdirectories like '00000', '00001', etc.
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
        self.num_classes = len(class_dirs)

        for class_name in class_dirs:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.ppm'):
                    self.samples.append((os.path.join(class_dir, file_name), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load the image
        image = imageio.imread(img_path)
        
        # Ensure image is 3-channel RGB
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]

        # ==> Apply your custom noise effect here! <==
        if self.effect:
            image = self.effect.apply(image) # Or self.effect(image) thanks to __call__

        # Apply standard torchvision transforms (e.g., ToTensor)
        if self.transform:
            image = self.transform(image)

        return image, label

# ====================================================================================
#  2. SIMPLE CONVOLUTIONAL NEURAL NETWORK MODEL
# ====================================================================================

class SimpleCNN(nn.Module):
    """A simple CNN for GTSRB classification."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Input: 3x32x32 -> Output: 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # Output: 16x16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# Output: 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # Output: 32x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# Output: 64x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # Output: 64x4x4
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ====================================================================================
#  3. TRAINING AND EVALUATION FUNCTIONS
# ====================================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

# ====================================================================================
#  4. MAIN EXECUTION SCRIPT
# ====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a GTSRB classifier with custom noise.")
    parser.add_argument("--data-dir", type=str, default="GTSRB/Final_Training/Images/", help="Path to GTSRB training data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--save-path", type=str, default="gtsrb_model.pth", help="Path to save the trained model.")
    parser.add_argument("--train-effect", type=str, default="none",
                        choices=['none', 'light_rain', 'heavy_rain', 'light_dust', 'heavy_dust'],
                        help="Noise effect to apply to the training set.")
    parser.add_argument("--eval-effect", type=str, default="none",
                        choices=['none', 'light_rain', 'heavy_rain', 'light_dust', 'heavy_dust'],
                        help="Noise effect to apply to the evaluation set.")
    args = parser.parse_args()

    # --- Define Effect Presets ---
    effect_presets = {
        "none": None,
        "light_rain": Rain(num_drops=15, radius_range=(2, 4)),
        "heavy_rain": Rain(num_drops=100, radius_range=(4, 8), magnification=1.15),
        "light_dust": Dust(num_specks=1000, num_scratches=1, splotch_opacity=0.1),
        "heavy_dust": Dust(num_specks=5000, num_scratches=5, splotch_opacity=0.25)
    }

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_effect_instance = effect_presets[args.train_effect]
    eval_effect_instance = effect_presets[args.eval_effect]

    # --- Data Loading ---
    # Standard transforms to be applied after the custom effect
    standard_transforms = transforms.Compose([
        transforms.ToTensor(), # This converts the numpy array (H,W,C) to a tensor (C,H,W) and scales to [0,1]
        transforms.Resize((32, 32), antialias=True),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]) # GTSRB normalization constants
    ])

    # We instantiate two separate datasets to assign different effects
    train_dataset = GTSRBDataset(root_dir=args.data_dir, transform=standard_transforms, effect=train_effect_instance)
    eval_dataset = GTSRBDataset(root_dir=args.data_dir, transform=standard_transforms, effect=eval_effect_instance)
    
    num_classes = train_dataset.num_classes
    print(f"Found {num_classes} classes and {len(train_dataset)} total images.")

    # Split the indices, then create subsets. This ensures both datasets use the same images.
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size])

    # Create subsets with the correct indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(eval_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

    # --- Model, Loss, Optimizer ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- Training Loop ---
    best_val_acc = 0.0
    print(f"\n--- Starting Training ---")
    print(f"Training with effect: '{args.train_effect}'")
    print(f"Evaluating with effect: '{args.eval_effect}'")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved to {args.save_path} with accuracy: {val_acc:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
