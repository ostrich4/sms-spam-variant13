from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, field_validator


# Репозиторий с твоей обученной моделью
REPO_ID = "sdsffdsasd/sms-spam-variant13"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Скачиваем модель из Hugging Face при запуске сервера
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="best_model.joblib"
    )

    # Загружаем TF-IDF и классификатор
    app.state.model = joblib.load(model_path)

    print("Модель успешно загружена из Hugging Face")

    yield


app = FastAPI(
    title="Классификация спама в SMS",
    description="Лабораторная работа №5. Вариант 13.",
    version="1.0",
    lifespan=lifespan
)


# Формат входящего запроса
class SMSRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Текст SMS на английском языке"
    )

    @field_validator("text")
    @classmethod
    def check_text(cls, value):
        if not value.strip():
            raise ValueError("Текст сообщения не должен быть пустым")
        return value


# Формат ответа
class SMSResponse(BaseModel):
    spam: int
    probability: float


@app.get("/")
def home():
    return {
        "status": "ok",
        "task": "SMS spam classification",
        "variant": 13
    }


@app.post("/predict", response_model=SMSResponse)
def predict(message: SMSRequest):
    model = app.state.model

    # Определяем класс: 0 — обычное сообщение, 1 — спам
    prediction = int(model.predict([message.text])[0])

    # Получаем вероятность именно класса spam = 1
    spam_index = list(model.classes_).index(1)
    probability = float(
        model.predict_proba([message.text])[0, spam_index]
    )

    return {
        "spam": prediction,
        "probability": round(probability, 8)
    }