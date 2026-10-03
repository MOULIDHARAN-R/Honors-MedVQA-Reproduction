import torch
from torchvision.utils import save_image
from src.config import CFG
from src.dataset import build_datasets

def main():
    # 1. Load the training dataset which contains the active augmentation pipeline
    print("Loading dataset...")
    train_ds, test_ds, vocab = build_datasets(CFG, 1)
    
    # 2. Fetch 16 augmented images
    num_samples = 16
    images = []
    for i in range(num_samples):
        # The dataset returns a dictionary; we only need the "image" tensor
        images.append(train_ds[i]["image"])
        
    image_tensor = torch.stack(images)  # Shape: [16, 3, 224, 224]
    
    # 3. Un-normalize the images (reverse the ImageNet mean/std to make them viewable)
    mean = torch.tensor(CFG.imagenet_mean).view(1, 3, 1, 1)
    std = torch.tensor(CFG.imagenet_std).view(1, 3, 1, 1)
    unnormalized_images = image_tensor * std + mean
    
    # 4. Save the tensor as a 4x4 image grid
    output_path = "augmented_samples.png"
    save_image(unnormalized_images, output_path, nrow=4)
    print(f"Successfully saved {num_samples} augmented samples to {output_path}")

if __name__ == "__main__":
    main()