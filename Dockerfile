FROM python:3.9
RUN mkdir /fastapi_app
WORKDIR /fastapi_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD gunicorn main:app --timeout 900 --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
