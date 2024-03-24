import datetime
import pandas as pd
from pydantic import BaseModel
from redis import asyncio as aioredis
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

from preprocessing import predicting_label
from config import REDIS_HOST, REDIS_PORT


class PredictionRequest(BaseModel):
    text: str


class ResponseText(BaseModel):
    hub: str


app = FastAPI()


@app.get("/")
@cache(expire=30)
async def root():
    """
    Приветственное сообщение
    """
    return "Welcome to hubs prediction service of Habr.ru"


@app.get("/ping")
@cache(expire=10)
async def ping():
    """
    Проверка доступности сервиса
    """
    now = datetime.datetime.now()
    return f"Service is running. Reply from {now}"


@app.post("/predict_text", response_model=ResponseText)
async def predict_text(request: PredictionRequest):
    """
    Предсказание подходящего хаба для введенного текста
    """
    output = predicting_label(request.text)
    return ResponseText(hub=output)


@app.post("/predict_text_from_txt", response_model=ResponseText)
async def predict_text_from_file(txt_file: UploadFile):
    """
    Предсказание подходящего хаба для текста из txt файла
    """
    text = await txt_file.read()
    text = text.decode('utf-8')
    output = predicting_label(text)
    return ResponseText(hub=output)


@app.post("/predict_texts_from_csv", response_class=FileResponse)
async def predict_texts_from_file(csv_file: UploadFile):
    """
    Предсказание подходящих хабов для текстов из csv файла,
    которые записываются в столбец 'hubs'
    """
    df = pd.read_csv(csv_file.file)
    df['hubs'] = df.iloc[:, 0].apply(predicting_label)
    df.to_csv('response.csv', index=False)
    new_name = csv_file.filename.split('.')[0]+'_predicted.csv'
    return FileResponse(
        'response.csv',
        filename=new_name,
        media_type="application/csv"
    )


@app.on_event("startup")
async def startup_event():
    """
    Инициализация Redis.
    """
    redis = aioredis.from_url(
        f"redis://{REDIS_HOST}:{REDIS_PORT}",
        encoding="utf8",
        decode_responses=True
    )
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
