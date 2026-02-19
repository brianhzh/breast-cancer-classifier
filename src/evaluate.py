import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay
from dataset import get_loaders
from model import CNN

def collect_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            probs = f.softmax(output, dim=1) # convert to probabilities

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return all_labels, all_probs

def plot_roc(labels, probs, classes):
    malignant_probs = probs[:, 1] # probability of malignant
    fpr, tpr, _ = roc_curve(labels, malignant_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray") # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, classes = get_loaders(batch_size=64)

    model = CNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("cnn_breakhis.pth", map_location=device))
    print(f"classes: {classes}")

    labels, probs = collect_predictions(model, test_loader, device)
    preds = np.argmax(probs, axis=1) # predicted class

    # metrics
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=classes))

    plot_roc(labels, probs, classes)
    plot_confusion_matrix(labels, preds, classes)

if __name__ == "__main__":
    main()
