from jetml.dataset.neural_network import IODataset
from jetml.utils import array_to_vector
from struct import unpack
import gzip
from array import array

def read(imagesfile, labelsfile):
    images = gzip.open(imagesfile, "rb")
    labels = gzip.open(labelsfile, "rb")

    images.read(4)
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    labels.read(4)
    number_of_labels = labels.read(4)
    number_of_labels = unpack('>I', number_of_labels)[0]

    if number_of_images != number_of_labels:
        raise Exception("The number of images does not equal the number of labels")
    
    dataset = IODataset()
    dataset_labels = [[], ["is 0", "is 1", "is 2", "is 3", "is 4", "is 5", "is 6", "is 7", "is 8", "is 9"]]
    for row in range(rows):
        for col in range(cols):
            dataset_labels[0].append("Pixel " + str(row) + "," + str(col))
    dataset.label = dataset_labels
    for i in range(number_of_images):
        pixels = []
        for row in range(rows):
            for col in range(cols):
                pixel = images.read(1)
                pixel = unpack(">B", pixel)[0]
                pixels.append(pixel)
        label = labels.read(1)
        label = unpack(">B", label)[0]
        output_l = [0] * 10
        output_l[label] = 1
        dataset.add(array_to_vector(pixels), array_to_vector(output_l))
    
    return dataset

