import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    def __init__(self, pet_dir, mri_dir, transform=None):
        """
        Args:
            pet_dir (string): Directory with all the FDG-PET images.
            mri_dir (string): Directory with all the MRI images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pet_dir = pet_dir
        self.mri_dir = mri_dir
        self.transform = transform

        # Assuming each folder corresponds to a patient and contains corresponding images
        self.patients = os.listdir(pet_dir)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pet_path = os.path.join(self.pet_dir, self.patients[idx], f"{self.patients[idx]}-dg1.gif")
        mri_path_1 = os.path.join(self.mri_dir, self.patients[idx], f"{self.patients[idx]}-mr1.gif")
        mri_path_2 = os.path.join(self.mri_dir, self.patients[idx], f"{self.patients[idx]}-mr2.gif")

        pet_image = Image.open(pet_path).convert('RGB')
        mri_image_1 = Image.open(mri_path_1).convert('RGB')
        mri_image_2 = Image.open(mri_path_2).convert('RGB')

        if self.transform:
            pet_image = self.transform(pet_image)
            mri_image_1 = self.transform(mri_image_1)
            mri_image_2 = self.transform(mri_image_2)

        return {'pet': pet_image, 'mri1': mri_image_1, 'mri2': mri_image_2}

# Usage Example
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = MedicalImageDataset(pet_dir='C://Users//sid55//Desktop//DDFM//MMIF-DDFM//dataset//fdgpet', mri_dir='C://Users//sid55//Desktop//DDFM//MMIF-DDFM//dataset//mri', transform=transform)
