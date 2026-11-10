import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Налаштування
CLASSES = ["yes", "no", "up", "down"]

DATASET_DIR = "./dataset/yes"
MODEL_DIR = "./models"
SAMPLE_RATE = 16000
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження власного датасету
class SpeechCommandsDataset(Dataset):
    def __init__(self, files):
        self.files = files

        self.class_to_idx = {
            class_name: i
            for i, class_name in enumerate(CLASSES)
        }
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path, label = self.files[index]
        waveform, sample_rate = torchaudio.load(file_path)
        # Перетворюємо стерео на моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Змінюємо частоту дискретизації за потреби
        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Вирівнюємо довжину до 1 секунди
        if waveform.shape[1] < SAMPLE_RATE:
            waveform = F.pad(
                waveform,
                (0, SAMPLE_RATE - waveform.shape[1])
            )
        else:
            waveform = waveform[:, :SAMPLE_RATE]

        # Mel-Spectrogram
        spectrogram = self.mel_transform(waveform)

        # Фіксуємо ширину спектрограми
        if spectrogram.shape[2] < 100:
            spectrogram = F.pad(
                spectrogram,
                (0, 100 - spectrogram.shape[2])
            )
        else:
            spectrogram = spectrogram[:, :, :100]
        target = self.class_to_idx[label]
        return spectrogram, target

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(12800, num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Підготовка списку аудіофайлів
def load_file_list():
    files = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(class_dir, filename)
                files.append((file_path, class_name))
    return files

# Основна програма
if __name__ == "__main__":
    print("Пристрій:", DEVICE)
    print("\nЗавантаження списку аудіофайлів...")
    files = load_file_list()
    print("Знайдено аудіофайлів:", len(files))
    # Розподіл 80% на навчання, 20% на тестування
    train_files, test_files = train_test_split(
        files,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in files]
    )

    print("Навчальна вибірка:", len(train_files))
    print("Тестова вибірка:", len(test_files))

    train_dataset = SpeechCommandsDataset(train_files)
    test_dataset = SpeechCommandsDataset(test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Створення моделі
    model = AudioCNN(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Навчання
    print("\nПочаток навчання моделі...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_loader)
        print(
            f"Епоха [{epoch + 1}/{EPOCHS}], "
            f"Loss: {average_loss:.4f}"
        )

    # Тестування моделі
    print("\nТестування моделі...")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nТочність на тестовій вибірці: {accuracy:.2f}%")

    # Збереження моделі
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(
        model.cpu().state_dict(),
        os.path.join(MODEL_DIR, "model.pth")
    )
    print("\nМодель збережено у ./models/model.pth")
