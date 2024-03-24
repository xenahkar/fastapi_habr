![image](https://github.com/xenahkar/fastapi_habr/assets/144525678/7976e8ea-2744-4f4d-b2e4-2fc4151a72a1)
# Веб-сервис для предсказания хабов на Хабре

 В рамках домашнего задания по курсу "Прикладной Python" реализован веб-сервис с использованием фреймворка FastAPI, который развернут через docker.

В ML части использовалась ранее предобученная модель из файла `linearSVC.pkl`, которая предсказывает подходящий хаб (aka тематика) для текста публикации. К текстам перед предсказанием применялась предобработка (удаление лишних символов и т.п., лемматизация с помощью библиотеки `pymystem3`). Для получившихся лемматизированных текстов производилась векторизация текстов по словам с помощью `tfidfvectorizer.pkl`. Упомянутые piсkle-файлы загружались из Yandex Cloud.

Автор: Карнакова Ксения (МОВС23)
 

 ## Реализованные методы

`/` - приветственное сообщение на главной странице

`/ping` - проверка доступности сервера

`/predict_text` - предсказание подходящего хаба для одного введенного текста

`/predict_text_from_txt` - предсказание подходящего хаба для одного текста из txt-файла

`/predict_texts_from_csv` - предсказание подходящих хабов для текстов из  csv-файла (в csv-файл добавляется новая колонка с предсказанием хаба для каждой статьи)

<img width="1000" alt="fastapi-habr-img" src="https://github.com/xenahkar/fastapi_habr/assets/144525678/df570504-3192-4b61-9bf1-14e3e27602d6">



## Запуск

```
docker-compose -f docker-compose.yml up
```

## Видео-демонстрация работы сервиса



https://github.com/xenahkar/fastapi_habr/assets/144525678/b795b00a-3386-49e3-b724-14d3a60afe17




Сервис на render: https://fastapi-habr.onrender.com


