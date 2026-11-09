import time
import torch
import torchaudio.transforms as T
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
import io

app = FastAPI(title="Audio Classification API", version="1.0")

CLASSES = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000

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
        self.fc = nn.Linear(32 * 32 * 25, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Завантажуємо модель при старті додатку
model = AudioCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("./models/model.pth", map_location=torch.device('cpu')))
model.eval()

transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)

@app.get("/")
def read_root():
    return {"message": "Audio MLOps Inference API is running inside Docker."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Читаємо аудіофайл у пам'ять
    audio_bytes = await file.read()
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    
    # Ресемплінг якщо потрібно
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        
    # Усереднення каналів до моно та приведення довжини
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    else:
        waveform = waveform[:, :SAMPLE_RATE]
        
    spec = transform(waveform).unsqueeze(0) # Додаємо батч-розмірність [1, 1, n_mels, time]
    
    if spec.shape[3] < 100:
        spec = torch.nn.functional.pad(spec, (0, 100 - spec.shape[3]))
    else:
        spec = spec[:, :, :, :100]

    with torch.no_grad():
        outputs = model(spec)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000.0

    predicted_label = CLASSES[predicted_idx.item()]

    return {
        "prediction": predicted_label,
        "confidence": round(confidence.item(), 4),
        "latency_ms": round(latency_ms, 2)
    }