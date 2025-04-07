import numpy as np
from skimage import io
import skimage.segmentation as seg
import skimage.color as color
from skimage.feature import canny
from skimage import transform
from matplotlib import pyplot as plt
from PIL import Image as img

# Загрузка изображения
image = io.imread('iimage.jpg')

# Преобразование изображения в двумерный массив (пиксели)
pixel_values = image.reshape((-1, 3))

# Применение K-средних для сегментации (например, 3 кластера)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixel_values)

# Получение сегментированных изображений
segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)

# Визуализация оригинального и сегментированного изображений
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Сегментированное изображение (K-средние)')
plt.axis('off')

plt.show()

# Загрузка изображения и преобразование в оттенки серого
gray_image = color.rgb2gray(io.imread('rgb_iimage.jpg'))

# Применение детектора Кэнни для выделения контуров
edges = feature.canny(gray_image)

# Визуализация оригинального и контурного изображений
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Оригинальное изображение (оттенки серого)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Контуры (детектор Кэнни)')
plt.axis('off')

plt.show()

# Применение детектора Кэнни для выделения контуров (можно использовать предыдущее изображение)
edges_hough = feature.canny(gray_image)

# Применение преобразования Хафа для поиска линий
hough_lines = transform.hough_line(edges_hough)

# Поиск пиков в преобразовании Хафа
h, theta, d = hough_lines
peaks = transform.hough_line_peaks(h, theta, d)

# Визуализация найденных линий на оригинальном изображении
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(gray_image, cmap='gray')
for _, angle in zip(*peaks):
    y0 = (d - np.cos(angle) * np.arange(-1000, 1000)) / np.sin(angle)
    ax.plot(np.arange(-1000, 1000), y0, '-r')
ax.set_title('Найденные линии (преобразование Хафа)')
ax.axis('off')
plt.show()