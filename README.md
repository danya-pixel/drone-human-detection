# drone-human-detection

Репозиторий по работе с YoloV8-nano для задачи детекции людей с дронов. 

## Установка и зависимости
Рекомендуемая версия `Python 3.10.9`. 

Сборка окружения:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Скачивание и препроцессинг датасета:

```
python3 src/data/download_data.py
```

В качестве датасета использовалась часть [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) для задачи детекции по изображениям (task 1). Для обучения на задачу детекции людей, первые два класса 

- 0: pedestrian
- 1: people

Были объеденены в один класс. Модель обучалась на одноклассовую детекцию. 

Для трекинга обучения использовался `comet_ml`. Результаты экспериментов и метрики можно также увидеть в [проекте](comet.com/danya-pixel/human-detection).


## Обучение моделей

Были обучены 3 версии YoloV8-n с разным размером изображений (imgsz): 640px, 840px, 1050px. 

Тестировал гипотезу, что с увеличением разрешения точность модели растет из-за специфики задачи детекции с дрона – объекты обычно очень маленькие относительно размера изображения. На VisDrone увеличение разрешения действительно приводит к повышению качества модели. Однако, работа в большем разрешении требует больше вычислительных ресурсов. 

Скачивание предобученных моделей:

| [YoloV8-n 640px](https://www.comet.com/api/asset/download?assetId=ef67432c5d3e41c5923ee74ce93c32e5&experimentKey=22ed95853eaf4a4da7b362850f53ac9d) | [ YoloV8-n 840px ](https://www.comet.com/api/asset/download?assetId=495e78c0e73a4e36bb4453beecc7a2ce&experimentKey=f73257b166c54bdab491af152551a67a)| [ YoloV8-n 1050px ](https://www.comet.com/api/asset/download?assetId=19f5b7dc65364d8a80eb35fd2db391c8&experimentKey=202b164634924872a6c9c3cd2bcf08da)|

---
Для воспроизведения обучения необходимо запустить тренировочный скрипт:

Для модели YoloV8-n 640px (default):
```
python3 src/models/train_model.py 
```

Для модели YoloV8-n 840px:

```
python3 src/models/train_model.py --batch 16 --imgsz 840
```

Для модели YoloV8-n 1050px:

```
python3 src/models/train_model.py --batch 12 --imgsz 1050
```

## Метрики и валидация
Итоговые метрики на валидации:

| Model        | Resolution | Images | Instances | Box(P) | R      | mAP50 | mAP50-95 |
|--------------|------------|--------|-----------|--------|--------|-------|----------|
| YoloV8-n     | 640px      | 548    | 13969     | 0.658  | 0.453  | 0.515 | 0.215    |
| YoloV8-n     | 840px      | 548    | 13969     | 0.715  | 0.545  | 0.615 | 0.277    |
| YoloV8-n     | 1050px     | 548    | 13969     | 0.759  | 0.581  | 0.664 | 0.312    |


Воспроизведение метрик на валидационном датасете. 

```
yolo detect val model="model_name.pth" data=datasets/cfg/VisDrone.yaml 
```


## Инференс модели
Скрипт для запуска инференса на фото/видео:
```
python3 src/models/predict_model.py --model_path="path_to_model.pth" --source "path_to_source"
```

либо командой:

```
yolo detect predict model="model_name.pth" source="path_to_source"
```


Для инференса на видео в лайв режиме:
```
python3 src/models/predict_live.py --model_path="path_to_model.pth" --video_path "path_to_video"
```

Также, есть возможность инференса с [sahi](https://github.com/obss/sahi) для улучшения качества на изображениях/видео высокого разрешения. 

```
python3 src/models/predict_sahi.py --model_path "model.pth" --soruce "source_path" --device cuda
```
## Результаты и ограничения модели


