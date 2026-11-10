import os
import csv
import time
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Налаштування
CLASSES = ["yes", "no", "up", "down"]
DATASET_DIR = "./dataset/yes"

API_URL = "http://localhost:8000/predict"

# Формуємо список аудіофайлів
def load_file_list():
    files = []

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(class_dir, filename)
                files.append((file_path, class_name))

    return files


if __name__ == "__main__":

    print("Завантаження списку аудіофайлів...")

    files = load_file_list()

    print("Знайдено аудіофайлів:", len(files))

    # Відтворюємо той самий поділ, що й під час навчання
    train_files, test_files = train_test_split(
        files,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in files]
    )

    print("Тестова вибірка:", len(test_files))
    print("\nПочаток тестування REST API...\n")

    true_labels = []
    predicted_labels = []
    api_latencies = []
    client_latencies = []

    results = []

    for i, (file_path, true_label) in enumerate(test_files):

        try:
            start_time = time.perf_counter()

            with open(file_path, "rb") as audio_file:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            os.path.basename(file_path),
                            audio_file,
                            "audio/wav"
                        )
                    },
                    timeout=30
                )

            client_latency = (
                time.perf_counter() - start_time
            ) * 1000

            response.raise_for_status()

            result = response.json()

            prediction = result["prediction"]
            api_latency = result["latency_ms"]

            true_labels.append(true_label)
            predicted_labels.append(prediction)

            api_latencies.append(api_latency)
            client_latencies.append(client_latency)

            results.append({
                "file": os.path.basename(file_path),
                "true_label": true_label,
                "prediction": prediction,
                "correct": true_label == prediction,
                "confidence": result["confidence"],
                "api_latency_ms": api_latency,
                "client_latency_ms": round(client_latency, 2)
            })

        except Exception as error:
            print("\nПомилка для файлу:", file_path)
            print(error)
            raise

        # Виводимо прогрес кожні 100 файлів
        if (i + 1) % 100 == 0 or (i + 1) == len(test_files):
            print(
                f"Оброблено: {i + 1}/{len(test_files)}"
            )

    # Розрахунок метрик класифікації
    accuracy = accuracy_score(true_labels, predicted_labels)

    print("\n========== РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ ==========")

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=CLASSES,
            target_names=CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
            labels=CLASSES
        )
    )

    # Розрахунок затримки
    api_latencies.sort()
    client_latencies.sort()

    average_api_latency = sum(api_latencies) / len(api_latencies)
    average_client_latency = sum(client_latencies) / len(client_latencies)

    p95_index = int(0.95 * len(api_latencies)) - 1
    p95_api_latency = api_latencies[p95_index]

    print("\n========== ПРОДУКТИВНІСТЬ API ==========")

    print(f"Середня затримка API: {average_api_latency:.2f} мс")
    print(f"P95 затримки API: {p95_api_latency:.2f} мс")
    print(f"Середній час HTTP-запиту: {average_client_latency:.2f} мс")

    # Зберігаємо результати у CSV
    os.makedirs("./results", exist_ok=True)

    with open(
        "./results/api_evaluation.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        fieldnames = [
            "file",
            "true_label",
            "prediction",
            "correct",
            "confidence",
            "api_latency_ms",
            "client_latency_ms"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nРезультати збережено у results/api_evaluation.csv")