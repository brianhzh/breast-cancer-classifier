import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BreakHisDataset(Dataset): # modify dataset for patient splitting (avoid leakage)
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_patient_id(filepath): # extract patient id from path
    parts = Path(filepath).parts
    for part in parts:
        if part.startswith("SOB_"): # patient directory
            return part
    return None

def collect_images(data_dir): # walk data dir, group images by patient
    class_to_label = {"benign": 0, "malignant": 1}
    patient_images = {}

    for class_name, label in class_to_label.items():
        class_dir = os.path.join(data_dir, class_name)
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if not fname.endswith(".png"):
                    continue
                filepath = os.path.join(root, fname)
                patient_id = get_patient_id(filepath)
                if patient_id is None:
                    continue
                if patient_id not in patient_images:
                    patient_images[patient_id] = []
                patient_images[patient_id].append((filepath, label))

    return patient_images

def split_by_patient(patient_images, train_ratio=0.8, seed=42):
    random.seed(seed)

    # separate patients by class to stratify
    benign_patients = [p for p, imgs in patient_images.items() if imgs[0][1] == 0]
    malignant_patients = [p for p, imgs in patient_images.items() if imgs[0][1] == 1]

    random.shuffle(benign_patients)
    random.shuffle(malignant_patients)

    # split each class separately
    b_split = int(len(benign_patients) * train_ratio)
    m_split = int(len(malignant_patients) * train_ratio)

    train_patients = set(benign_patients[:b_split] + malignant_patients[:m_split])
    test_patients = set(benign_patients[b_split:] + malignant_patients[m_split:])

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for patient_id, imgs in patient_images.items():
        paths = [p for p, _ in imgs]
        labels = [l for _, l in imgs]
        if patient_id in train_patients:
            train_paths.extend(paths)
            train_labels.extend(labels)
        else:
            test_paths.extend(paths)
            test_labels.extend(labels)

    return train_paths, train_labels, test_paths, test_labels

def get_loaders(data_dir='C:/Users/brian/BCC/data', batch_size=64): # load data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # converting to tensor and standardizing data
    ])

    train_transform = transforms.Compose([ # more data
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patient_images = collect_images(data_dir)
    train_paths, train_labels, test_paths, test_labels = split_by_patient(patient_images)

    print(f"patients: {len(patient_images)} ({len([p for p,imgs in patient_images.items() if imgs[0][1]==0])} benign, {len([p for p,imgs in patient_images.items() if imgs[0][1]==1])} malignant)")
    print(f"train: {len(train_paths)} images | test: {len(test_paths)} images")

    train_dataset = BreakHisDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = BreakHisDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True) # nonrandom test data

    classes = ["benign", "malignant"]
    return train_loader, test_loader, classes
