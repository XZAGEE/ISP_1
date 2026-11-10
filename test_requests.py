import requests

url = "http://localhost:8000/predict"

print("Надсилання запиту до API у Docker...")
try:
    with open("file_example_WAV_10MG.wav", "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        print("Результат відповіді сервера:")
        print(response.json())
except FileNotFoundError:
    print("Будь ласка, покладіть будь-який файл 'sample.wav' у корінь проєкту для тестування запиту.")