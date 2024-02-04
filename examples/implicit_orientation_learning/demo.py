import os
import json
import argparse
import yaml
import numpy as np
import cv2
import pickle
from tensorflow.keras.utils import get_file
# from sklearn.metrics.pairwise import euclidean_distances as measure
from sklearn.metrics.pairwise import cosine_similarity as measure
from paz.backend.camera import VideoPlayer, Camera
from paz.backend.image import load_image, show_image
from scenes import DictionaryView

from model import AutoEncoder
from pipelines import ImplicitRotationPredictor


parser = argparse.ArgumentParser(description='Implicit orientation demo')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-f', '--y_fov', type=float, default=3.14159 / 4.0,
                    help='field of view')
parser.add_argument('-v', '--viewport_size', type=int, default=128,
                    help='Size of rendered images')
parser.add_argument('-d', '--distance', type=float, default=1000.0,
                    help='Distance between camera and 3D model')
parser.add_argument('-s', '--shift', type=float, default=0.01,
                    help='Shift')
parser.add_argument('-l', '--light', type=int, default=2,
                    help='Light intensity') # 2 for powerdrill, 2.7 for ape
parser.add_argument('-b', '--background', type=int, default=0,
                    help='Plain background color')
parser.add_argument('-r', '--roll', type=float, default=3.14159,
                    help='Maximum roll')
parser.add_argument('-t', '--translate', type=float, default=0.01,
                    help='Maximum translation')
parser.add_argument('-p', '--top_only', type=int, default=0,
                    help='Rendering mode')
parser.add_argument('--theta_steps', type=int, default=20,
                    help='Amount of steps taken in the X-Y plane')
parser.add_argument('--phi_steps', type=int, default=20,
                    help='Amount of steps taken from the Z-axis')
parser.add_argument('--model_name', type=str,
                    default='SimpleAutoencoder128_128_035_power_drill',
                    help='Model directory name without root')
parser.add_argument('--model_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models/'),
                    help='Root directory PAZ trained models')
args = parser.parse_args()


path = os.path.join(args.model_path, args.model_name)
parameters = json.load(open(os.path.join(path, 'hyperparameters.json'), 'r'))

size = parameters['image_size']
latent_dimension = parameters['latent_dimension']
weights_path = os.path.join(path, args.model_name + '_weights.hdf5')
K_real = np.array([
    [572.41140, 000.00000, 5.26110],
    [000.00000, 573.57043, 2.04899],
    [000.00000, 000.00000, 001.00000]],
    dtype=np.float32)

fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
linemod_camera_params = [fx, fy, px, py]
# obj_path = get_file('textured.obj', None,
#                     cache_subdir='paz/datasets/ycb/models/035_power_drill/')
obj_path = 'obj_08.ply'
cam_W, cam_H = 640, 480
renderer = DictionaryView(
    obj_path, (cam_W, cam_H), args.y_fov,
    args.distance, bool(args.top_only), args.light, args.theta_steps,
    args.phi_steps)
dict_images = renderer.render()
for i in range(len(dict_images)):
    img = dict_images[i]['image']
    cv2.imwrite('dict_images/img_{}.png'.format(i), img)


encoder = AutoEncoder((size, size, 3), latent_dimension, mode='encoder')
encoder.load_weights(weights_path, by_name=True)
decoder = AutoEncoder((size, size, 3), latent_dimension, mode='decoder')
decoder.load_weights(weights_path, by_name=True)
inference = ImplicitRotationPredictor(encoder, decoder, measure, renderer)
#0817, 0114, 0133 doesnt work
IMAGE_PATH = ('/home/manummk95/Desktop/paz/paz/examples/efficientpose/'
              'Linemod_preprocessed/data/08/rgb/0000.png')
anno_path = ('/home/manummk95/Desktop/paz/paz/examples/efficientpose/'
             'Linemod_preprocessed/data/08/gt.yml')
f_real = (fx + fy) / 2.0

with open(anno_path, 'r') as f:
    file_contents = yaml.safe_load(f)
    f.close()
# IMAGE_PATH = 'SimpleAutoencoder128_128_035_power_drill/original_images/image_010.png'
anno_key = int(os.path.split(IMAGE_PATH)[1].split('.')[0])
bbox = file_contents[anno_key][0]['obj_bb']
x_min, y_min, W, H = bbox
x_max = x_min + W
y_max = y_min + H

# ####### Compute focal length of pyrender camera ###
f_syn = cam_H / (2 * np.tan(args.y_fov / 2))
t_syn = args.distance
image = load_image(IMAGE_PATH)
image = image[y_min:y_max, x_min:x_max]
image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
output = inference(image, t_syn, f_syn, f_real, [x_min, y_min, x_max, y_max], K_real)
print('Real z :{}'.format(file_contents[anno_key][0]['cam_t_m2c']))
print('Estimated z :{}'.format(output['t_real_z']))
print('Treal :{}'.format(output['t_reals']))
show_image(output['image'])

output['image_path'] = IMAGE_PATH
# Write outputs ###
with open('pose_pred.pkl', 'wb') as f:
    pickle.dump(output, f)

# player = VideoPlayer((1280, 960), inference, camera=Camera(args.camera_id))
# player.run()
