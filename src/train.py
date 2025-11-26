import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import CNN
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels) # forward pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # backpropagation
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = output.max(1) # not too sure what the _, does but i need it i guess
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() # track stats

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def eval(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = output.max(1) 
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # track stats

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, classes = get_loaders(batch_size=64)

    model = CNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0003) # Adam optimizer, added weight decay

    num_epochs = 25
    # lr decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0018,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )   
     
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_acc = eval(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.2f} | Test  Acc: {test_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}") # monitor progress

    torch.save(model.state_dict(), "cnn_breakhis.pth")
    print("done")

if __name__ == "__main__":
    main()
    