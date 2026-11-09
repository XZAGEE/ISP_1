import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

# Визначаємо класи та параметри
CLASSES = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

class SubsetSpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root, download, subset_classes):
        super().__init__(root, download=download)
        # Фільтруємо датасет лише за потрібними класами
        self._walker = [
            p for p in self._walker 
            if os.path.basename(os.path.dirname(p)) in subset_classes
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(subset_classes)}

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        target = self.class_to_idx[label]
        return waveform, sample_rate, target

# Проста згорткова мережа для спектрограм
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(32 * 32 * 25, num_classes) # адаптується під розмір спектрограми

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def collate_fn(batch):
    tensors, targets = [], []
    transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
    
    for waveform, sample_rate, target in batch:
        # Уніфікація довжини аудіо (рівно 1 секунда = 16000 семплів)
        if waveform.shape[1] < SAMPLE_RATE:
            waveform = torch.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
        else:
            waveform = waveform[:, :SAMPLE_RATE]
            
        spec = transform(waveform)
        # Зменшуємо часову шкалу до фіксованого розміру (наприклад, 100 кадрів)
        if spec.shape[2] < 100:
            spec = torch.pad(spec, (0, 100 - spec.shape[2]))
        else:
            spec = spec[:, :, :100]
            
        tensors.append(spec)
        targets.append(target)
        
    return torch.stack(tensors), torch.tensor(targets)

if __name__ == "__main__":
    print("Завантаження датасету...")
    train_data = SubsetSpeechCommands(root="./dataset", download=True, subset_classes=CLASSES)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = AudioCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Початок навчання моделі...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Епоха [{epoch+1}/{EPOCHS}], Крок [{i}], Втрати (Loss): {loss.item():.4f}")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/model.pth")
    print("Модель успішно навчена та збережена у ./models/model.pth")