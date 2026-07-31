#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_shape(self, points, name, difficult, is_polygon=False):
        self.box_list.append({
            'points': points,
            'name': name,
            'difficult': difficult,
            'is_polygon': is_polygon
        })

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.add_shape(points, name, difficult, is_polygon=False)

    def bnd_box_to_yolo_line(self, box, class_list=[]):
        # Kept for compatibility if called externally, but we handle it manually in save()
        pts = box['points']
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=[], target_file=None):

        out_file = None  # Update yolo .txt
        out_class_file = None   # Update class list .txt

        if target_file is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classes_file, 'w')

        else:
            out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")
            out_class_file = open(classes_file, 'w')


        for box in self.box_list:
            box_name = box['name']
            if box_name not in class_list:
                class_list.append(box_name)
            class_index = class_list.index(box_name)

            if box.get('is_polygon', False):
                coords = []
                for p in box['points']:
                    x_norm = float(p[0]) / self.img_size[1]
                    y_norm = float(p[1]) / self.img_size[0]
                    coords.append(f"{x_norm:.6f} {y_norm:.6f}")
                out_file.write(f"{class_index} " + " ".join(coords) + "\n")
            else:
                class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
                out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        unique_classes = []
        for c in class_list:
            c_str = str(c).strip()
            if c_str and c_str not in unique_classes:
                unique_classes.append(c_str)

        for c in unique_classes:
            out_class_file.write(c + '\n')

        out_class_file.close()
        out_file.close()



class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        # print (file_path, self.class_list_path)

        classes_file = open(self.class_list_path, 'r')
        self.classes = classes_file.read().strip('\n').split('\n')

        # print (self.classes)

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = False
        # try:
        self.parse_yolo_format()
        # except:
        #     pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):

        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult, False))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        label = self.classes[int(class_index)]

        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        bnd_box_file = open(self.file_path, 'r')
        for bndBox in bnd_box_file:
            parts = bndBox.strip().split(' ')
            if len(parts) == 5:
                class_index, x_center, y_center, w, h = parts
                label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)

                # Caveat: difficult flag is discarded when saved as yolo format.
                self.add_shape(label, x_min, y_min, x_max, y_max, False)
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                class_index = int(parts[0])
                label = self.classes[class_index]
                raw_points = []
                for i in range(1, len(parts), 2):
                    x_norm = float(parts[i])
                    y_norm = float(parts[i+1])
                    x = round(x_norm * self.img_size[1])
                    y = round(y_norm * self.img_size[0])
                    raw_points.append((x, y))
                
                # Deduplicate consecutive identical points exported by external web tools!
                clean_points = []
                for pt in raw_points:
                    if not clean_points or pt != clean_points[-1]:
                        clean_points.append(pt)

                if len(clean_points) >= 3:
                    self.shapes.append((label, clean_points, None, None, False, True))
