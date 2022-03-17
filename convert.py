# import gzip
# f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 10

import idx2numpy
import numpy as np
file = 'train-images.idx3-ubyte'
data = idx2numpy.convert_from_file(file)
data = data[:10,:,:]

print(data.shape)

from PIL import Image

for i, d in enumerate(data):
    img = Image.fromarray(d, 'L')
    img.save(f"./images/{i}.png")