import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from train import SimpleCNN  # Reuse the model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Setting up transforms...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10 test dataset...")
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False)

print("Loading trained model...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model/cnn_cifar10.pth", map_location=device))
model.eval()  # Switch to evaluation mode
print("Model loaded and set to eval mode.")


print("Starting evaluation...")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
