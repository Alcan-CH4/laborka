import numpy as np
from math import pi
from skimage import io
from skimage import morphology
from skimage import measure
from skimage import filters
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image as img

# Загрузка изображения и преобразование в оттенки серого
image = io.imread('iimage.jpg')
gray_image = color.rgb2gray(image)

# Ручной подбор порога
manual_threshold = 0.5
binary_manual = gray_image > manual_threshold

# Подбор порога методом Оцу
otsu_threshold = filters.threshold_otsu(gray_image)
binary_otsu = gray_image > otsu_threshold

# Построение гистограммы яркостей
plt.figure(figsize=(12, 6))
plt.hist(gray_image.ravel(), bins=256, histtype='step', color='black')
plt.axvline(manual_threshold, color='red', linestyle='--', label='Ручной порог')
plt.axvline(otsu_threshold, color='blue', linestyle='--', label='Порог Оцу')
plt.title('Гистограмма яркостей')
plt.xlabel('Яркость')
plt.ylabel('Частота')
plt.legend()
plt.show()

# Выполнение морфологических операций
eroded_image = binary_erosion(binary_otsu)
dilated_image = binary_dilation(binary_otsu)
closed_image = binary_closing(binary_otsu)
opened_image = binary_opening(binary_otsu)

# Визуализация результатов
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(eroded_image, cmap='gray')
axes[0].set_title('Эрозия')
axes[1].imshow(dilated_image, cmap='gray')
axes[1].set_title('Дилатация')
axes[2].imshow(closed_image, cmap='gray')
axes[2].set_title('Замыкание')
axes[3].imshow(opened_image, cmap='gray')
axes[3].set_title('Размыкание')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

# Сегментация методом связных компонент
labelled_image = measure.label(binary_otsu)
regions = measure.regionprops(labelled_image)

# Визуализация сегментированных областей и их свойств
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(labelled_image, cmap='nipy_spectral')

for region in regions:
    # Рисуем прямоугольник вокруг каждой области
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(minc + 5, minr + 5, f'Area: {region.area}', color='white')

ax.set_title('Сегментация связных компонент')
ax.axis('off')
plt.show()

# Предполагаем наличие RGB-изображения для распознавания объектов.
rgb_image = io.imread('rgb_iimage.jpg')

# Применяем бинаризацию (например с использованием метода Оцу)
binary_rgb = rgb_image[:, :, 0] > otsu_threshold  # Используем один канал для бинаризации

# Сегментация и получение свойств объектов в бинарном изображении
labelled_rgb = measure.label(binary_rgb)
regions_rgb = measure.regionprops(labelled_rgb)

# Визуализация распознанных объектов на RGB-изображении
fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8))
ax_rgb.imshow(rgb_image)

for region in regions_rgb:
    # Рисуем прямоугольник вокруг каждого объекта
    minr, minc, maxr, maxc = region.bbox
    rect_rgb = plt.Rectangle((minc, minr), maxc - minc,
                             maxr - minr,
                             fill=False,
                             edgecolor='red',
                             linewidth=2)
    ax_rgb.add_patch(rect_rgb)
    ax_rgb.text(minc + 5, minr + 5,
                 f'Area: {region.area}',
                 color='white')

ax_rgb.set_title('Распознавание объектов на RGB-изображении')
ax_rgb.axis('off')
plt.show()