import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Transformations ===
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# === DenseNet121 Model ===
def load_densenet_model(class_count):
    model = models.densenet121(weights='IMAGENET1K_V1')
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, class_count)
    return model.to(device)

# === Training Function ===
def train_model(train_loader, test_loader, class_names, model, optimizer, criterion, epochs=30):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_loss / len(test_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

# === Evaluation Function ===
def evaluate_model(loader, model, dataset_type="Test"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\nAccuracy on {dataset_type} Set: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# === Plotting Function ===
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title("Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Main Loop for Training Days ===
for day in ['Day 3', 'Day 4', 'Day 5']:
    print(f"\n\nTraining for {day}...")

    TRAIN_PATH = f'/content/drive/MyDrive/embryo images/Train/{day}'
    TEST_PATH = f'/content/drive/MyDrive/embryo images/Test/{day}'

    if not os.path.exists(TRAIN_PATH):
        print(f"Skipping {day} - Train folder not found.")
        continue

    if not os.path.exists(TEST_PATH):
        print(f"Skipping {day} - Test folder not found.")
        TEST_PATH = TRAIN_PATH

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = train_dataset.classes

    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    # Using DenseNet121 Model
    model = load_densenet_model(len(class_names))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        train_loader, test_loader, class_names, model, optimizer, criterion
    )

    print(f"\n👉 Finished training for {day}. If satisfied, save manually using:\n"
          f"torch.save(model.state_dict(), '{day.replace(' ', '_')}_densenet121_model.pth')")

    evaluate_model(train_loader, model, f"Train ({day})")
    evaluate_model(test_loader, model, f"Test ({day})")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # === Save to Summary CSV ===
    summary_file = "/content/drive/MyDrive/embryo images/final_model_performance_summary.csv"
    summary_row = {
        "Day": day,
        "Model": "densenet121",
        "Final Train Loss": round(train_losses[-1], 4),
        "Final Train Accuracy (%)": round(train_accuracies[-1], 2),
        "Final Val Loss": round(val_losses[-1], 4),
        "Final Val Accuracy (%)": round(val_accuracies[-1], 2)
    }

    if os.path.exists(summary_file):
        df_existing = pd.read_csv(summary_file)
        df_new = pd.concat([df_existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([summary_row])

    df_new.to_csv(summary_file, index=False)
    print(f"✅ Results for {day} (DenseNet121) saved to {summary_file}")
