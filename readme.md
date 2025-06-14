# Краткий README.md для проекта

## Danbooru Image Tagger API

Простой веб-сервис для автоматического тегирования изображений с помощью машинного обучения.

## Что это

API принимает изображения и возвращает список тегов с помощью efficientnetv2_s модели. Модель из timm.

## Запуск

### Docker (рекомендуется)
```bash
docker build -t danbooru-tagger .
docker run -d -p 8000:8000 danbooru-tagger
```

### Локально
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Использование

### Проверка работы
```bash
curl http://localhost:8000/
```

### Тегирование изображения
```bash
curl -X POST "http://localhost:8000/tags" -F "file=@image.jpg"
```

## API

- `GET /` - проверка статуса
- `POST /tags` - загрузка изображения и получение тегов

## Структура проекта

```
app/
├── main.py              # FastAPI приложение
└── services/
    └── model_service.py # ML сервис
models/
└── best_model.pth      # Обученная модель
requirements.txt        # Зависимости
Dockerfile             # Docker конфигурация
```