import sys
from fastapi.testclient import TestClient
from freezegun import freeze_time
from functools import wraps
from unittest import mock


# мокер на кэш
def mock_cache(*args, **kwargs):
    def wrapper(func):
        @wraps(func)
        async def inner(*args, **kwargs):
            return await func(*args, **kwargs)
        return inner
    return wrapper


# имитируем работу кеша
mock.patch("fastapi_cache.decorator.cache", mock_cache).start()

sys.path.insert(1, './')
from main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to hubs prediction service of Habr.ru"


@freeze_time("2024-03-20 00:00:00")
def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == f"Service is running. Reply from 2024-03-20 00:00:00"


def test_predict_text():
    response = client.post(
        "/predict_text",
        json={"text": "string"},
    )

    assert response.status_code == 200


def test_predict_text_from_file():

    file_path = "./tests/test_txt_text.txt"
    files = {"txt_file": open(file_path, "rb")}
    response = client.post("/predict_text_from_txt", files=files)
    files["txt_file"].close()

    assert response.status_code == 200


def test_predict_texts_from_file():

    file_path = "./tests/test_csv_texts.csv"
    files = {"csv_file": open(file_path, "rb")}
    response = client.post("/predict_texts_from_csv", files=files)
    assert response.status_code == 200
