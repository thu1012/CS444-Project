import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MAEDataset(Dataset):
    def __init__(self, base_image_path, txt_path, is_train=True):
        self.base_image_path = base_image_path
        
        # get txt
        with open(txt_path, 'r') as f:
            self.image_ids = [line.strip().split()[0] for line in f.readlines()]
        
        # get all paths
        self.image_paths = [
            os.path.join(base_image_path, f"{img_id}.jpg")  # jpg
            for img_id in self.image_ids
            if os.path.exists(os.path.join(base_image_path, f"{img_id}.jpg"))
        ]
        
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),              # Resize
                transforms.RandomHorizontalFlip(),      # Flip
                transforms.ToTensor(),                  # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),               # Resize
                transforms.ToTensor(),                  # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        image = self.transform(image)  # Apply transformations
        return image


class Classify_Dataset(Dataset): 
    def __init__(self, base_image_path, txt_path, is_train=True):
        self.base_image_path = base_image_path

        # Parse txt file to get image_id and class_id
        with open(txt_path, 'r') as f:
            self.image_ids = {
                line.strip().split()[0]: int(line.strip().split()[1])  # {image_id: class_id}
                for line in f.readlines()
            }

        # Get all valid image paths
        self.image_paths = [
            os.path.join(base_image_path, f"{img_id}.jpg")  # jpg
            for img_id in self.image_ids.keys()
            if os.path.exists(os.path.join(base_image_path, f"{img_id}.jpg"))
        ]

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),             # Resize
                transforms.RandomHorizontalFlip(),         # Flip
                transforms.ToTensor(),                    # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),             # Resize
                transforms.ToTensor(),                    # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]  # Extract image_id
        class_id = self.image_ids[image_id]  # Get corresponding class_id
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        image = self.transform(image)  # Apply transformations
        return image, class_id


def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
