
import os
import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom

XML_PATH = 'VOCdevkit/VOC2007/Annotation/'

def ConvertAnnotations(img:np.ndarray, annotation: np.ndarray) -> np.ndarray:
    """
     Convert XML  (xmin, ymin, xmax, ymax) annotations to YOLO (x,y,w,h).

    :param img: Image read with cv2 and stored as a NumPy array
    :param annotation: Entries from .xml files represented as NumPy arrays.
    :return: array of object localization in YOLO annotations.
    """

    HEIGHT, WIDTH = img.shape

    xmin = 0
    xmax = 0

    ymin = 0
    ymax = 0

    x_yolo = 0
    y_yolo = 0
    w = 0
    h = 0

    obj_loc = np.array([x_yolo, y_yolo, w, h])

    return obj_loc

def XML2NumPy(xml):
    """
        Convert XML file to NumPy array.
    :param xml: Path to XML files
    :return: Object localization in NumPy format
    """

    localizations = minidom.parse(xml)

    file_name = localizations.getElementsByTagName('filename')
    file_name = file_name[0].firstChild.data

    size = localizations.getElementsByTagName('size')
    height = size[0].getElementsByTagName("height")[0].firstChild.data
    width = size[0].getElementsByTagName("width")[0].firstChild.data

    print(width)
    #size = file_name

    #print(size)

    # Number of objects
    Nobj = 1


    obj_loc_voc = np.zeros((Nobj, 4))

    return obj_loc_voc

def ReadDirectory():
    """
        Read .jpeg files and convert them to a TensorFlow Dataset format.
    :return: TensorFlow Dataset
    """

    for filename in os.listdir('Annotations'):

        xml = os.path.join('Annotations/', filename)
        XML2NumPy(xml)

        break


def main():
    ReadDirectory()


if __name__ == '__main__':
    main()