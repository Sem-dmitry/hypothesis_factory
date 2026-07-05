# Фабрика гипотез

Промышленный мульти-агентный сервис для генерации, проверки и ранжирования научно-технических гипотез в обогащении руд и металлургии.

Система принимает цель, KPI, ограничения и корпус данных предприятия: Excel-таблицы, PDF/DOCX-отчеты, регламенты, схемы оборудования и PNG-изображения. Далее она строит частный RAG-корпус, подтягивает открытые источники, генерирует гипотезы, проверяет их агентами Reflection/Evolution/Meta-review, ранжирует через ELO-турнир и формирует готовые артефакты для инженеров.

Каждая гипотеза содержит механизм влияния, причинно-следственную связь, промышленную применимость, соблюдение ограничений, новизну относительно входных данных, KPI-эффект, риски, экономическую и кинетическую оценку, план проверки и точные ссылки на источники вплоть до Excel-листа/строк, страницы PDF или изображения.

Киллер-фичи: автоуточнение недостающих ограничений до генерации, работа с мультимодальными производственными данными, экспертная доработка гипотез, мультиагентный ELO-турнир и готовность к полностью закрытому контуру.

## Основной интерфейс

Основной интерфейс проекта - актуальная FastAPI SPA в `webapp/`, а не legacy Streamlit.

```bash
pip install -e .
pip install -r requirements.txt
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8800
```

Откройте:

```text
http://127.0.0.1:8800
```

В интерфейсе можно создать проект, задать цель и ограничения, загрузить корпус файлов, включить или выключить веб-исследование, наблюдать работу агентов, смотреть гипотезы, отчет, метрики, граф источников и скачивать полный лог запуска.

## Настройка `.env`

Скопируйте пример и заполните ключ RouterAI:

```bash
cp .env.example .env
```

Минимальный `.env`:

```dotenv
ROUTER_AI_API_KEY=your-routerai-key
ROUTER_AI_BASE_URL=https://routerai.ai/api/v1
ROUTER_AI_EMBEDDING_MODEL=text-embedding-3-small
ROUTER_AI_EMBEDDING_DIMENSIONS=256
```

RouterAI используется как OpenAI-compatible endpoint для chat, vision, embeddings и web-search. Если отдельные embedding-переменные не заданы, система переиспользует `ROUTER_AI_API_KEY` и `ROUTER_AI_BASE_URL`.

Дополнительные настройки:

```dotenv
# Переопределение моделей
COSCIENTIST_DEFAULT_MODEL=claude-sonnet-4-20250514
COSCIENTIST_VISION_MODEL=google/gemini-2.5-pro
COSCIENTIST_WEBSEARCH_MODEL=google/gemini-2.5-flash

# JSON-маппинг локальных алиасов на provider model id
ROUTER_AI_MODEL_MAP={"o3":"openai/o3","gemini-2.5-pro":"google/gemini-2.5-pro"}

# Отдельный endpoint/key для embeddings, если нужен
ROUTER_AI_EMBEDDING_API_KEY=your-routerai-embedding-key
ROUTER_AI_EMBEDDING_BASE_URL=https://routerai.ai/api/v1

# Хранилище индексов и запусков
COSCIENTIST_DIR=~/.coscientist
```

PowerShell:

```powershell
Copy-Item .env.example .env
$env:ROUTER_AI_API_KEY = "your-routerai-key"
$env:ROUTER_AI_BASE_URL = "https://routerai.ai/api/v1"
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8800
```

## Быстрый запуск без webapp

Офлайн-демо без сетевых вызовов:

```bash
python scripts/run_pipeline.py --offline-demo --out out
```

Реальный запуск по локальному корпусу:

```bash
python scripts/run_pipeline.py \
  --data-dir data \
  --goal "Снизить потери никеля с хвостами флотации без потери качества концентрата" \
  --out out
```

Артефакты сохраняются в `out/`: отчет, задачи, JSON, граф и вспомогательные файлы.

## Корпус и RAG

Корпус можно загрузить через webapp или построить индекс отдельно:

```bash
python -m coscientist.corpus.build --data-dir data
```

Без разбора изображений через vision model:

```bash
python -m coscientist.corpus.build --data-dir data --no-images
```

Поддерживаются PDF, DOCX, XLSX/Excel, PNG/JPEG и текстовые материалы. Каждый chunk хранит структурную ссылку: файл, лист/строки Excel, страницу PDF/DOCX или изображение. Эти ссылки попадают в гипотезы и финальные объяснения.

## Как работает пайплайн

1. Constraints agent уточняет недостающие ограничения и допущения.
2. Literature agent собирает контекст из частного корпуса и, если включено, из веб-источников через RouterAI web-search.
3. Generation agent формирует гипотезы с учетом цели, KPI, хвостов, минералогии, оборудования и ограничений.
4. Reflection/Evolution/Meta-review проверяют и улучшают гипотезы.
5. Ranking agent проводит ELO-турнир.
6. Assessor структурирует итоговую оценку, риски, экономику, кинетику, KPI-эффект и план проверки.
7. Final report agent формирует инженерный отчет.

## Тесты

```bash
python -m pytest -q
```

Полезные точечные проверки:

```bash
python -m pytest tests/test_model_factory.py tests/test_web_search.py tests/test_webapp.py -q
```

## Основные директории

- `webapp/` - актуальный FastAPI/static интерфейс.
- `coscientist/` - мультиагентный пайплайн, RAG, модели, web-search, assessment и экспорт.
- `coscientist/corpus/` - загрузчики PDF/DOCX/XLSX/изображений и построение индекса.
- `coscientist/prompts/` - системные промпты агентов.
- `scripts/` - CLI-запуск пайплайна.
- `tests/` - офлайн-регрессии без реальных сетевых вызовов.
