import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

"""
# K-means
from sklearn.cluster import KMeans

km = KMeans(init='random', n_clusters=3, random_state=42)
km.fit(fruits_2d)
"""

# K-means ++
from sklearn.cluster import KMeans

km = KMeans(init='k-means++', n_clusters=3, random_state=42)
km.fit(fruits_2d)

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr) # sample 개수
    # 한 줄에 10개씩 이미지를 그린다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산한다.
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수이다. else 10개
    cols = n if rows < 2 else 10
    fig, axs = plt.subplot(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i,j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()