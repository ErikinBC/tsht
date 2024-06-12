"""
Get the MNIST data
"""

from mnist import MNIST
mndata = MNIST('./dir_with_mnist_data_files')
for i in range(10):
    print(i)
    b = i+2