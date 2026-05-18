# YOLO12 Number Recognition System

Система распознавания стартовых номеров участников на фотографиях соревнований.

Проект состоит из четырёх частей:

- `frontend` — тестовый web-клиент на статическом HTML/CSS/JS + Nginx;
- `backend` — API на Go + Gin;
- `ml_service` — FastAPI-сервис инференса YOLO-моделей;
- `postgres` — база данных для задач и результатов.

## Быстрый ответ по GPU

Видеокарта для работы проекта не обязательна: backend, frontend и PostgreSQL работают без GPU, а ML-сервис может выполнять инференс на CPU. GPU нужен только для ускорения распознавания и особенно полезен при большом количестве фотографий. Подробности по переносу см. в [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Возможности

- загрузка одного или нескольких изображений через web-интерфейс;
- создание фоновой задачи распознавания;
- сохранение статуса задачи и результата в PostgreSQL;
- получение результата по `task_id`;
- вывод найденных номеров и confidence-метрик;
- вывод обработанного изображения с bbox найденного номера;
- опциональный callback/webhook после завершения задачи;
- локальный запуск backend + frontend + PostgreSQL через Docker Compose;
- серверный Compose-вариант для запуска frontend + backend + PostgreSQL + ML service.

## Структура проекта

```text
yolo12-number-system/
  backend/
    cmd/server/main.go
    internal/
    migrations/
    Dockerfile
  frontend/
    index.html
    styles.css
    app.js
    nginx.conf
    Dockerfile
  ml_service/
    app/main.py
    app/recognizer.py
    models/*.pt
    Dockerfile
  tools/
    callback_receiver.py
  docker-compose.yml
  docker-compose.server.yml
  .env.server.example
```

## API backend

Все защищённые методы требуют заголовок:

```http
X-API-Key: super-secret-key
```

### Проверка backend

```http
GET /api/status
```

Ответ:

```json
{"status":"ok"}
```

### Создание задачи по URL

```http
POST /api/tasks
Content-Type: application/json
X-API-Key: super-secret-key
```

```json
{
  "images": [
    {
      "id": "image-1.jpg",
      "url": "https://example.com/image-1.jpg"
    }
  ]
}
```

Опционально можно передать callback:

```json
{
  "images": [
    {
      "id": "image-1.jpg",
      "url": "https://example.com/image-1.jpg"
    }
  ],
  "callback": {
    "url": "http://host.docker.internal:9100/callback",
    "mode": "result"
  }
}
```

`callback.mode`:

- `result` — отправить полный результат задачи;
- `status` — отправить только сообщение о готовности.

### Загрузка файлов

```http
POST /api/tasks/upload
Content-Type: multipart/form-data
X-API-Key: super-secret-key
```

Поле для файлов:

```text
files
```

Можно передать один или несколько файлов.

Пример через PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/api/tasks/upload" `
  -H "X-API-Key: super-secret-key" `
  -F "files=@C:\path\to\image.jpg"
```

С callback:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/api/tasks/upload" `
  -H "X-API-Key: super-secret-key" `
  -F "files=@C:\path\to\image.jpg" `
  -F "callback_url=http://host.docker.internal:9100/callback" `
  -F "callback_mode=result"
```

### Получение результата задачи

```http
GET /api/tasks/{task_id}
X-API-Key: super-secret-key
```

### Получение изображения с bbox

```http
GET /api/task-images/{task_image_id}/annotated
X-API-Key: super-secret-key
```

Web-клиент загружает это изображение через `fetch` с `X-API-Key`, поэтому endpoint остаётся защищённым.

## Локальный запуск

Локальный режим рассчитан на то, что `ml_service` запускается на Windows/host-машине через существующее Python-окружение с CUDA, а `frontend`, `backend` и `postgres` запускаются в Docker.

### 1. Запустить ML service

```powershell
cd "C:\Users\karpe\desktop\dz\ВКРБ\code\yolo12-number-system\ml_service"

C:\Users\karpe\desktop\dz\ВКРБ\code\yolo12-training\.venv\Scripts\Activate.ps1

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Проверка:

```powershell
curl.exe http://127.0.0.1:8000/health
```

### 2. Запустить frontend + backend + PostgreSQL

```powershell
cd "C:\Users\karpe\desktop\dz\ВКРБ\code\yolo12-number-system"

docker compose down
docker compose up -d --build
```

Проверка backend:

```powershell
curl.exe http://127.0.0.1:8080/api/status
```

Web-клиент:

```text
http://127.0.0.1:3000
```

В локальном `docker-compose.yml` backend обращается к ML service через:

```text
http://host.docker.internal:8000
```

## Серверный запуск

Серверный вариант поднимает все компоненты в Docker:

- `postgres`;
- `ml_service`;
- `backend`;
- `frontend`.

### 1. Создать `.env.server`

```bash
cp .env.server.example .env.server
```

Измени значения:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change-me
POSTGRES_DB=number_system
API_KEY=change-me-api-key
TASK_WORKERS=2
FRONTEND_PORT=80
```

### 2. Запустить серверный compose

```bash
docker compose --env-file .env.server -f docker-compose.server.yml up -d --build
```

Проверка:

```bash
curl http://127.0.0.1/api/status
```

Если на сервере используется NVIDIA GPU, установи NVIDIA Container Toolkit и раскомментируй в `docker-compose.server.yml` строку:

```yaml
gpus: all
```

## Callback receiver для теста webhook

Callback нужен не браузеру, а внешнему серверу, который должен получить результат без постоянного опроса.

Запуск тестового получателя:

```powershell
python .\tools\callback_receiver.py
```

URL для Docker-backend на локальной машине:

```text
http://host.docker.internal:9100/callback
```

## Проверка и сборка backend

```powershell
cd backend
go test ./...
```

## Что не надо коммитить

В репозиторий не должны попадать:

- `backend/.env`;
- `backend/uploads/`;
- `__pycache__/`;
- старые папки `backend_old/`, `ml_service_old/`, `test_client/`;
- архивы `.zip`.
