# Dog Breed Detection

Классификация пород собак с использованием Vision Transformer и PyTorch Lightning.

## Описание проекта

Проект решает задачу классификации пород собак по фотографиям. Оригинальное соревнование: [Dog Breed Identification (Kaggle)](https://www.kaggle.com/competitions/dog-breed-identification)

### Формат входных и выходных данных

- **Вход**: изображения собак в формате `.jpg/.png`
- **Выход**: вероятности для каждой из 120 пород собак

### Метрики

Основная метрика — **MultiClassLogLoss** (Cross-Entropy Loss) между предсказанными вероятностями и реальным классом. Так же считается f1-score и обычный accuracy.

### Датасет

Источник: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

- 20,580 фотографий собак
- 120 пород
- ~150 изображений на каждый класс
- Включает bounding boxes для каждого изображения

### Моделирование

- **Основная модель**: Fine-tuning Vision Transformer (ViT) на базе решения [dog-breed-0.17-loss-finetune-on-vit](https://www.kaggle.com/code/l066858998/dog-breed-0-17-loss-finetune-on-vit)

### Итоговые метрики

| Test metric   | DataLoader 0       |
| ------------- | ------------------ |
| test/accuracy | 0.763362467288971  |
| test/f1       | 0.7587199211120605 |
| test/loss     | 1.175703525543213  |

### Инференс

Для инференса в рамках проекта используется Triton Inference Server.

---

## Setup

### Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/your-username/dog-breed-detection.git
cd dog-breed-detection
```

2. Создайте виртуальное окружение и установите зависимости с помощью `uv`:

```bash
# Установите uv, если ещё не установлен
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создайте venv и установите зависимости
uv venv
source .venv/bin/activate

uv pip install -e ".[dev]"
```

3. Установите pre-commit хуки:

```bash
pre-commit install
```

4. Проверьте, что все хуки проходят:

```bash
pre-commit run -a
```

### Загрузка данных

Данные хранятся с помощью DVC. Для загрузки выполните:

```bash
dog-breed download_data
```

Или вручную:

```bash
dvc pull
```

---

## Train

### Мониторинг обучения

Запустите MLflow для просмотра экспериментов:

```bash
cd mlflow
cp .env.dev.example .env
docker compose up -d
```

Затем откройте http://127.0.0.1:8080 в браузере.

MLflow Tracking URL для логгирования задается в [mlflow.yaml](configs/logging/mlflow.yaml)

### Запуск обучения

Базовый запуск с конфигурацией по умолчанию:

```bash
dog-breed train
```

### Изменение гиперпараметров

Можно переопределить любые параметры через CLI:

```bash
dog-breed train --overrides "['training.epochs=50', 'training.batch_size=64', 'training.learning_rate=5e-5']"
```

### Экспорт модели

Произведите экспорт модели в формат поддерживаемый triton:

```bash
# экспорт заранее обученной модели
dog-breed export models/best-epoch02-valloss0.9876.ckpt --format=torchscript
# или экспорт обученной вами модели
dog-breed export models/{YOUR_CHECKPOINT_NAME}.ckpt --format=torchscript
```

> [!WARNING]
> Хоть скрипт может экспортировать модель в формат onnx, trtion последней версии не поддерживает версию библиотеки, используемую в проекте (смотрите [issue](https://github.com/triton-inference-server/server/issues/8001))

---

## Inference

### Предсказание для одного изображения

```bash
dog-breed infer_triton triton/dog.jpeg

2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Model dog_breed_classifier is ready
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Input shape: (1, 3, 224, 224)
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Running inference...
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | ==================================================
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Predicted breed: n02090379-redbone
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Confidence: 70.38%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Class index: 17
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | Top 5 predictions:
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands |   1. n02090379-redbone: 70.38%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands |   2. n02087394-Rhodesian_ridgeback: 13.26%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands |   3. n02107312-miniature_pinscher: 3.88%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands |   4. n02100583-vizsla: 2.41%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands |   5. n02099712-Labrador_retriever: 2.14%
2025-12-30 15:47:28 | INFO | dog_breed_detection.commands | ==================================================
```

---

## Структура проекта

```
dog-breed-detection/
├── configs/                          # Hydra конфигурации
│   ├── config.yaml                   # Главный конфиг
│   ├── data/
│   │   └── default.yaml              # Параметры данных
│   ├── model/
│   │   └── vit.yaml                  # Конфигурация ViT модели
│   ├── training/
│   │   └── default.yaml              # Параметры тренировки
│   └── logging/
│       └── mlflow.yaml               # Настройки MLflow
│
├── dog_breed_detection/              # Python пакет
│   ├── __init__.py
│   ├── commands.py                   # CLI команды (fire + hydra)
│   ├── data/                         # Загрузка и препроцессинг данных
│   │   ├── dataset.py                # PyTorch Dataset
│   │   ├── datamodule.py             # Lightning DataModule
│   │   └── transforms.py             # Аугментации
│   ├── models/                       # Архитектуры моделей
│   │   └── classifier.py             # ViT классификатор
│   ├── training/                     # Тренировка
│   │   ├── lightning_module.py       # PyTorch Lightning Module
│   │   └── train.py                  # Скрипт тренировки
│   ├── inference/                    # Инференс
│   │   └── predict.py                # Предсказание
│   └── utils/                        # Утилиты
│       ├── download.py               # Загрузка и распаковка данных
│       ├── export.py                 # Экспорт в ONNX/TorchScript
│       └── logging.py                # Конфигурация логирования
│
├── data/                             # Данные (под DVC)
│   ├── annotations/                  # Аннотации датасета
│   ├── images/                       # Изображения собак
│   ├── annotations.tar.gz.dvc        # DVC-файл для аннотаций
│   └── images.tar.gz.dvc             # DVC-файл для изображений
│
├── models/                           # Сохранённые модели и чекпоинты
│   ├── *.ckpt                        # PyTorch Lightning чекпоинты
│   └── *.onnx                        # Экспортированные ONNX модели
│
├── mlflow/                           # MLflow конфигурация
│   └── docker-compose.yml            # Docker Compose для MLflow сервера
│
├── triton/                           # NVIDIA Triton Inference Server
│   └── docker-compose.yml            # Docker Compose для Triton
│
├── class_mapping.json.dvc            # DVC-файл для маппинга индексов к названиям пород
├── .pre-commit-config.yaml           # Pre-commit хуки (ruff, prettier)
├── pyproject.toml                    # Зависимости и настройки (uv)
└── README.md                         # Документация
```
