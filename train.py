import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # [32x15x15]
            nn.Conv2d(32, 64, 3, 1),    # [64x13x13]
            nn.ReLU(),
            nn.MaxPool2d(2),            # [64x6x6]
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading CIFAR-10 training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()            # Zero the gradients
            outputs = model(images)          # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update weights

            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}", flush=True)

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/cnn_cifar10.pth")
    print("Training complete and model saved!")
