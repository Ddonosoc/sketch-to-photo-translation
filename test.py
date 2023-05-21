import cv2
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf


image_file = 'D:/Tesis/Models/Evaluation/ResNet50/experiments_results/diffusion/500/shoe/shoes/2437700078_2.png'
image = tf.io.read_file(image_file)
image = tf.image.decode_jpeg(image)
RGB = tf.reshape(image, (256 * 256, 3))
RGB = tf.cast(RGB, tf.int32)
Rhist = tf.histogram_fixed_width(RGB[:, 0], [0, 256], nbins=32)
Ghist = tf.histogram_fixed_width(RGB[:, 1], [0, 256], nbins=32)
Bhist = tf.histogram_fixed_width(RGB[:, 2], [0, 256], nbins=32)
hist = tf.concat([Rhist, Ghist, Bhist], 0)

# imageObj = cv2.imread(image_file)
# to avoid grid lines
# plt.axis("off")
# plt.title("Original Image")
# plt.bar(np.arange(0, len(hist)), hist, color="blue")

# # # plt.imshow(image)
# plt.show()

# blue_color = cv2.calcHist([imageObj], [0], None, [64], [0, 256])
# red_color = cv2.calcHist([imageObj], [1], None, [64], [0, 256])
# green_color = cv2.calcHist([imageObj], [2], None, [64], [0, 256])

# blue_color = list(map(lambda r: r[0], blue_color))
# hist = tf.histogram_fixed_width(blue_color, [0, 255], nbins=16).numpy()

# print(len(hist), len(blue_color), (hist - blue_color).sum())


# # Separate Histograms for each color
# plt.subplot(3, 1, 1)
# plt.title("histogram of Blue")
# plt.bar(np.arange(0, len(blue_color)), list(map(lambda r: r[0], green_color)), color="blue")

  
# plt.subplot(3, 1, 2)
# plt.title("histogram of Green")
# plt.bar(np.arange(0, len(blue_color)), list(map(lambda r: r[0], blue_color)), color="green")
  
# plt.subplot(3, 1, 3)
# plt.title("histogram of Red")
# plt.bar(np.arange(0, len(blue_color)), list(map(lambda r: r[0], red_color)), color="red")
  
# # for clear view
# plt.tight_layout()
# plt.show()
  
# combined histogram
# plt.title("Histogram of all RGB Colors")
# plt.bar(np.arange(0, len(Bhist)), Bhist, color="blue")
# plt.bar(np.arange(0, len(Ghist)), Ghist, color="green")
# plt.bar(np.arange(0, len(Rhist)), Rhist, color="red")
# plt.show()
