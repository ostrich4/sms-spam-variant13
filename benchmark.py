import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


# Можно указать локальный адрес или адрес облачного сервиса
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output", default="latency_local.json")
args = parser.parse_args()

url = args.url.rstrip("/") + "/predict"

payload = {
    "text": (
        "Free entry in 2 a wkly comp to win FA Cup final tkts. "
        "Text FA to 87121 now!"
    )
}

body = json.dumps(payload).encode("utf-8")


def send_request():
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")

        if result.get("spam") not in (0, 1):
            raise ValueError("Некорректный класс в ответе")

        probability = result.get("probability")

        if not isinstance(probability, (int, float)):
            raise ValueError("Вероятность не является числом")

        if not 0 <= probability <= 1:
            raise ValueError("Вероятность вне диапазона 0–1")

        return result


print("ПРОВЕРКА ПРОИЗВОДИТЕЛЬНОСТИ API")
print("Адрес:", url)

# Прогрев не включаем в измерения
print("\nПрогрев: 5 запросов...")

for _ in range(5):
    send_request()

times = []
errors = []
last_response = None

print("Выполняются 100 последовательных запросов...")

for i in range(100):
    start = time.perf_counter()

    try:
        result = send_request()
        elapsed_ms = (time.perf_counter() - start) * 1000

        times.append(elapsed_ms)
        last_response = result

    except Exception as error:
        errors.append({
            "request": i + 1,
            "error": str(error)
        })

metrics = {
    "url": url,
    "requests": 100,
    "warmup_requests": 5,
    "successful": len(times),
    "failed": len(errors),
    "mean_ms": float(np.mean(times)) if times else None,
    "p95_ms": float(np.percentile(times, 95)) if times else None,
    "min_ms": min(times) if times else None,
    "max_ms": max(times) if times else None,
    "last_response": last_response,
    "times_ms": times,
    "errors": errors
}

Path(args.output).write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nРЕЗУЛЬТАТЫ")
print("Всего измеряемых запросов:", metrics["requests"])
print("Успешных:", metrics["successful"])
print("Ошибок:", metrics["failed"])

if times:
    print(f"Средняя задержка: {metrics['mean_ms']:.2f} мс")
    print(f"p95: {metrics['p95_ms']:.2f} мс")
    print(f"Минимальная задержка: {metrics['min_ms']:.2f} мс")
    print(f"Максимальная задержка: {metrics['max_ms']:.2f} мс")
    print(
        "Последний ответ:",
        json.dumps(last_response, ensure_ascii=False)
    )

print("\nРезультаты сохранены в:", args.output)

if errors:
    print("Первая ошибка:", errors[0])
    raise SystemExit("Проверка завершилась с ошибками.")