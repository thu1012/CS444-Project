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

# Example usage
# train_dataset = MAEDataset(base_image_path='path/to/images', txt_path='path/to/trainval.txt', is_train=True)
# eval_dataset = MAEDataset(base_image_path='path/to/images', txt_path='path/to/test.txt', is_train=False)




def get_dataloader(base_path, batch_size=32, shuffle=True, num_workers=4):

    dataset = MAEDataset(base_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
