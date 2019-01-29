import cv2
import numpy as np
import matplotlib.pyplot as plt

resim = cv2.imread('resimler/dag.jpg')

# grayscale donusturme islemi
gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

gri_resim = np.float32(gri_resim)

# Harris kose tespit
kose = cv2.cornerHarris(gri_resim, 2, 3, 0.04)

# kose noktalarını arttırmak icin dilate
kose = cv2.dilate(kose, None)

threshold = 0.02 * kose.max()

resim_kopya  = np.copy(resim)

for j in range(0, kose.shape[0]):
    for i in range(0, kose.shape[1]):
        if(kose[j, i] > threshold):
            # resim, orta nokta, yaricap, renk, kalınlık
            cv2.circle(resim_kopya, (i, j), 2, (0,255,0), 1)
plt.imshow(resim_kopya, cmap='gray')
plt.show()


