import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

model_path = r"C:\Users\DELL\crop_disease_detection\evaluation\best_crop_model.pth"
test_dir = r"C:\Users\DELL\crop_disease_detection\evaluation\test_data\test"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(test_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

model = efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

start_time = time.time()

print(f"Starting evaluation on {len(dataset)} images...")
print(f"Total batches: {len(dataloader)}\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

  
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)} processed | Images done: {total}")

end_time = time.time()

print("\n Computing metrics...")

accuracy = correct / total
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print("Metrics computed")

print("Building confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
print("Confusion matrix done")

inference_time = (end_time - start_time) / total

print("\n========== FINAL RESULTS ==========")
print("Total images:", len(dataset))
print("Total processed:", total)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

print(f"\nAverage Inference Time per image: {inference_time * 1000:.2f} ms")

print("\nConfusion Matrix:")
print(cm)