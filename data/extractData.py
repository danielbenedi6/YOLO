
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from xml.dom import minidom
import pandas as pd
from xml.etree import ElementTree

XML_PATH = 'VOCdevkit/VOC2007/Annotation/'

def ConvertAnnotations(h, w, x1, x2, y1, y2) -> np.ndarray:
    """
     Convert XML  (xmin, ymin, xmax, ymax) annotations to YOLO (x,y,w,h).

    :param img: Image read with cv2 and stored as a NumPy array
    :param annotation: Entries from .xml files represented as NumPy arrays.
    :return: array of object localization in YOLO annotations.
    """
    #print(h, w, x1, x2, y1, y2)

    x_yolo = (x2 + x1) / (2 * w)
    y_yolo = (y2 + y1) // (2 * h)

    w_yolo = (x2 - x1) / w
    h_yolo = (y2 - y1) / h

    return (x_yolo, y_yolo, w_yolo, h_yolo)

def XML2NumPy(xml):
    """
        Convert XML file to NumPy array.
    :param xml: Path to XML files
    :return: Object localization in NumPy format
    """

    localizations = minidom.parse(xml)

    size = localizations.getElementsByTagName('size')
    height = size[0].getElementsByTagName("height")[0].firstChild.data
    width = size[0].getElementsByTagName("width")[0].firstChild.data

    file_name = localizations.getElementsByTagName('filename')
    bounding_boxes = localizations.getElementsByTagName('bndbox')
    Nobj = len(bounding_boxes)

    obj_loc_voc = np.zeros((Nobj, 4))



    from xml.etree import ElementTree as ET

    doc = ET.parse(xml)

    for ind, bb in enumerate(doc.findall(".//bndbox")):
        xmin = float(bb.findall(".//xmin")[0].text)
        ymin = float(bb.findall(".//ymin")[0].text)
        xmax = float(bb.findall(".//xmax")[0].text)
        ymax = float(bb.findall(".//ymax")[0].text)

        bb = (xmin, xmax, ymin, ymax)
        bb = ConvertAnnotations(int(height), int(width), *bb)
        #print(bb)
        obj_loc_voc[ind] = bb

    return obj_loc_voc


def jpg2NumPy(img_path):

    dim = (448,448)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return img


def ReadDirectory():
    """
        Read .jpeg files and convert them to a TensorFlow Dataset format.
    :return image_data: Images represented with NumPy matrices, shape = (Nimages, 448,448,3)
    :return bb_data: Bounding boxes in NumPy matrices stored in a list.
        The number of element corresponds to image in row of image_data
    """

    Nimages = len(os.listdir('Annotations'))

    image_data = np.zeros((Nimages, 448,448,3))

    bb_data = []


    for ind, filename in tqdm(enumerate(os.listdir('Annotations')), total = Nimages):

        xml = os.path.join('Annotations/', filename)

        image_name = filename[:-4] + '.jpg'
        jpg_file = os.path.join('Images/', image_name)

        bounding_boxes = XML2NumPy(xml)
        img = jpg2NumPy(jpg_file)

        image_data[ind] = img
        bb_data.append(bounding_boxes)

    return image_data, bb_data

def main():
    ReadDirectory()


if __name__ == '__main__':
    main()


""" Notes.
for box in bounding_boxes:
    box_nodes = box.childNodes

    for box_node in box_nodes:
        node_elements = box_node.childNodes

        for box_elem in node_elements:
            print(box_elem.data)
"""