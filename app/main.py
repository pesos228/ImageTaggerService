# app/main.py

from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .services.model_service import model_service

app = FastAPI(
    title="Danbooru Image Tagger API",
    description="API для определения тегов на изображениях с помощью ML-модели.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники. В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST и т.д.)
    allow_headers=["*"],
)


@app.get("/", summary="Health Check", description="Проверка доступности сервиса.")
async def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "Service is running successfully 🚀"}
    )

@app.post(
    "/tags",
    response_model=List[str],
    summary="Get Image Tags",
    description="Загрузите изображение для получения списка из 15 наиболее вероятных тегов."
)
async def get_image_tags(file: UploadFile = File(..., description="Файл изображения для анализа (JPEG, PNG).")):
    # Проверка типа файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload an image (JPEG, PNG)."
        )

    try:
        # Читаем содержимое файла в байты
        image_bytes = await file.read()
        
        # Передаем байты в наш сервис
        tags = model_service.predict_tags(image_bytes, top_n=15)
        
        return tags
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the image."
        )
