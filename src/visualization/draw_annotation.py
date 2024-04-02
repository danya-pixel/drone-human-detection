import os
import random

import cv2
import matplotlib.pyplot as plt


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Функция для отрисовки bounding box на изображении
    tl = (
        line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # Толщина линии/шрифта
    color = color or [random.randint(0, 255) for _ in range(3)]  # Случайный цвет
    c1, c2 = (int(x[0]), int(x[1])), (
        int(x[2]),
        int(x[3]),
    )  # Координаты углов прямоугольника
    cv2.rectangle(
        image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA
    )  # Отрисовка прямоугольника
    if label:  # Если есть метка
        tf = max(tl - 1, 1)  # Толщина шрифта
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
            0
        ]  # Размер текста
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # Координаты для текста
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # Прямоугольник для текста
        cv2.putText(
            image,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )  # Текст


def draw_boxes(image_name, classes, colors, label_folder, raw_image_folder):
    """
    Функция для добавления bounding box на изображения.
    """
    txt_path = os.path.join(
        label_folder, "%s.txt" % image_name
    )  # Путь к файлу с аннотациями
    if image_name == ".DS_Store" or not os.path.exists(txt_path):
        return None  # Если нет файла аннотации, пропустить
    image_path = os.path.join(
        raw_image_folder, "%s.jpg" % image_name
    )  # Путь к оригинальному изображению

    image = cv2.imread(image_path)  # Загрузка изображения
    height, width, channels = image.shape  # Размеры изображения

    with open(txt_path, "r") as file:
        for line in file:
            staff = line.split()  # Разбиение строки аннотации
            class_idx = int(staff[0])  # Индекс класса
            x_center, y_center, w, h = (
                float(staff[1]) * width,
                float(staff[2]) * height,
                float(staff[3]) * width,
                float(staff[4]) * height,
            )  # Координаты bounding box
            x1 = round(x_center - w / 2)
            y1 = round(y_center - h / 2)
            x2 = round(x_center + w / 2)
            y2 = round(y_center + h / 2)

            plot_one_box(
                [x1, y1, x2, y2],
                image,
                color=colors[class_idx],
                label=classes[class_idx],
            )  # Отрисовка bounding box

    return image


def show_images_in_grid(
    image_names,
    classes,
    colors,
    label_folder,
    raw_image_folder,
    grid_shape=(3, 3),
    figsize=(10, 10),
):
    """
    Функция для отображения изображений в виде сетки.
    """
    num_images = min(len(image_names), grid_shape[0] * grid_shape[1])
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image_name = image_names[i]
            image = draw_boxes(
                os.path.splitext(image_name)[0],
                classes,
                colors,
                label_folder,
                raw_image_folder,
            )
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(image_name)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_random_images(root_folder, classes, shape):
    """
    Функция для отображения случайных изображений в виде сетки.
    """
    label_folder = root_folder / "labels"
    raw_image_folder = root_folder / "images"
    image_names = os.listdir(raw_image_folder)
    random.shuffle(image_names)  # Перемешать список
    selected_image_names = image_names[
        : shape[0] * shape[1]
    ]  # Выбрать случайные изображения
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    show_images_in_grid(
        selected_image_names,
        classes,
        colors,
        label_folder,
        raw_image_folder,
        grid_shape=shape,
    )
