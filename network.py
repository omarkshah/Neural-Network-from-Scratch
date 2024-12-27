import numpy as np
import struct
from array import array

# Load the MNIST dataset
def read_images_labels(images_filepath, labels_filepath):        
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    
    return images, labels

x_train, y_train = read_images_labels("./data/train-images-idx3-ubyte/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
x_test, y_test = read_images_labels("./data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

