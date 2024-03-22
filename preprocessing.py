import re
import pickle
import nltk
import boto3
from nltk.corpus import stopwords
from pymystem3 import Mystem
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET


s3 = boto3.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

tfidfvectorizer = pickle.loads(s3.get_object(Bucket=BUCKET, Key="tfidfvectorizer.pkl").get('Body').read())
model = pickle.loads(s3.get_object(Bucket=BUCKET, Key="linearSVC.pkl").get('Body').read())


nltk.download('stopwords')
stopwords_russian = stopwords.words('russian')
stopwords_english = stopwords.words('english')


mystem = Mystem()


def remove_tags(text):
    """
    удаляет теги html
    """
    text = re.sub(r'<[^>]*>', "", text, flags=re.MULTILINE)
    return text


def remove_http(text):
    """
    удаляет ссылки и e-mail
    """
    text = re.sub(
        r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', " ",
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r"([\w+-]+[\w.+-]*@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)",
        " ",
        text,
        flags=re.MULTILINE
    )
    return text


def remove_space(text):
    """
    удаляет разделители и лишние пробелы
    """
    text = re.sub(r'[\s\n\t]', " ", text)
    text = re.sub(r'\s{2,}', " ", text)
    return text


def lemmatize_text(text):
    """
    Лемматизация текста
    если слова нет в списке стоп-слов,
    слово длиннее двух символов
    состоит только из букв/из букв и дефиса
    """
    text_tokens = mystem.lemmatize(text.lower())
    tokens = [
        token for token in text_tokens
        if token not in (stopwords_russian or stopwords_english)
        and len(token) > 2
           and (token.isalpha() or token.split('-')[0].isalpha())
    ]
    return " ".join(tokens)


def preprocessing_text(text):
    text = remove_tags(text)
    text = remove_http(text)
    text = remove_space(text)
    text = lemmatize_text(text)
    return text


def predicting_label(item: str) -> str:
    preprocessed_item = preprocessing_text(item)
    preprocessed_item = tfidfvectorizer.transform([preprocessed_item])
    predicted_label = model.predict(preprocessed_item)
    return predicted_label[0]
