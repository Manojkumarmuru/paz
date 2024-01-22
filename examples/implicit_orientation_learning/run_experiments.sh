#!/bin/bash
class_name="035_power_drill"
images_directory="VOCdevkit/VOC2007/JPEGImages/"
object_path="obj_08.ply"

python3 train.py --images_directory $images_directory --obj_path $object_path --class_name $class_name -st 4
