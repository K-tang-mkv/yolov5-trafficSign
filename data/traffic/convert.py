from os.path import isfile
import glob   
import os 
import xml.etree.ElementTree as ET

cls_names = ['red', 'yellow', 'green', 'turn_right', 'turn_left', 'stop']

dir = "./xml/"

# get all sorted xmls
list_xmls = sorted(filter(isfile, glob.glob(dir + '*')))

# get the name id of each image.xml
images_id = [f[6:-4] for f in list_xmls]

lbs_path = './labels'
if not os.path.isdir('./labels'):
    os.mkdir('./labels')

lb_path = [lbs_path + '/' + image_id + '.txt' for image_id in images_id]

for label in lb_path:
    open(label, 'a+') # create label.txt

def convert(lb_path, list_xmls, images_id):
    def convert_box(size, box):
          dw, dh = 1. / size[0], 1. / size[1]
          x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
          return x * dw, y * dh, w * dw, h * dh
    
    for image_id in range(len(images_id)):
        in_file = open(list_xmls[image_id])
        out_file = open(lb_path[image_id], 'w')
        
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if not int(obj.find('difficult').text) == 1:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w,h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = cls_names.index(cls) # class name id
                out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

if __name__ == '__main__':
    convert(lb_path, list_xmls, images_id)
