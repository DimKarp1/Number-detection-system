# Deployment guide

## 1. Компоненты системы

Проект состоит из четырёх контейнеризуемых частей:

- `frontend` — статический тестовый клиент на Nginx;
- `backend` — Go + Gin API;
- `postgres` — база данных задач и результатов;
- `ml_service` — FastAPI + YOLO inference.

## 2. Нужна ли видеокарта

Для работы всей системы видеокарта не обязательна.

- `frontend`, `backend` и `postgres` работают без GPU.
- `ml_service` может выполнять инференс YOLO на CPU.
- GPU нужен не для запуска проекта как такового, а для ускорения распознавания.

Режимы:

```text
CPU only  — проект работает, но распознавание медленнее.
NVIDIA GPU — распознавание быстрее, полезно для большого числа фото.
```

Для другой Windows-машины без NVIDIA GPU можно запустить backend/frontend/postgres в Docker, а ML-сервис — через Python venv или Docker CPU. Для демонстрации и небольшого потока изображений CPU обычно достаточно.

Для обучения моделей GPU практически необходим, но обучение не входит в обычный runtime-перенос проекта.

## 3. Перенос на другую Windows-машину

### Требования

- Docker Desktop;
- Git, если проект переносится через репозиторий;
- Python 3.11, если ML-сервис запускается через venv, а не через Docker;
- свободные порты: `3000`, `8080`, `55432`, `8000`.

### Вариант A: локальный режим как сейчас

В этом режиме в Docker запускаются:

- postgres;
- backend;
- frontend.

ML-сервис запускается отдельно через Python venv.

```powershell
cd yolo12-number-system\ml_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ML_DEVICE="auto"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

В другом PowerShell:

```powershell
cd yolo12-number-system
docker compose up -d --build
```

Проверки:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8080/api/status
```

Frontend:

```text
http://127.0.0.1:3000
```

## 4. Серверный запуск через Docker Compose

Скопировать env:

```bash
cp .env.server.example .env
```

Отредактировать `.env`:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=strong-password
POSTGRES_DB=number_system
API_KEY=strong-api-key
FRONTEND_PORT=80
ML_DEVICE=auto
```

Запуск:

```bash
docker compose -f docker-compose.server.yml --env-file .env up -d --build
```

Проверка:

```bash
docker compose -f docker-compose.server.yml ps
curl http://127.0.0.1/api/status
```

## 5. Сервер с NVIDIA GPU

GPU не обязателен. Если на сервере есть NVIDIA GPU и нужно ускорить inference:

1. Установить NVIDIA driver на host.
2. Установить NVIDIA Container Toolkit.
3. В `.env` указать:

```env
ML_DEVICE=cuda:0
```

4. В `docker-compose.server.yml` у `ml_service` раскомментировать:

```yaml
gpus: all
```

Важно: Docker-образ ML-сервиса должен содержать CUDA-совместимую сборку PyTorch. Если контейнер собран с CPU-only PyTorch, `ML_DEVICE=cuda:0` не заработает, даже если на сервере есть видеокарта.

## 6. Сервер без GPU

Оставить:

```env
ML_DEVICE=cpu
```

или:

```env
ML_DEVICE=auto
```

Такой вариант проще и переносимее, но медленнее на большом количестве фотографий.

## 7. Полезные команды

Логи всех сервисов:

```bash
docker compose -f docker-compose.server.yml logs -f
```

Логи ML-сервиса:

```bash
docker compose -f docker-compose.server.yml logs -f ml_service
```

Пересборка:

```bash
docker compose -f docker-compose.server.yml up -d --build
```

Остановка:

```bash
docker compose -f docker-compose.server.yml down
```

Остановка с удалением БД и runtime-volume:

```bash
docker compose -f docker-compose.server.yml down -v
```
