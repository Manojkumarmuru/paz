import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene
from pyrender import RenderFlags, Mesh, Scene, viewer
import trimesh
import cv2


class SingleView():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        light: List of floats indicating [max_light, min_light]
        top_only: Boolean. If True images are only take from the top.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """
    def __init__(self, filepath, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(filepath, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0],
                           ambient_light=[0.1255, 0.1255, 0.1255, 1.0])
        self.light = self.scene.add(
            DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(trimesh.load(path), smooth=True))
        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        theta = np.random.uniform(np.pi/2, 0)
        x = distance * np.sin(theta) * np.cos(0)
        y = distance * np.sin(theta) * np.sin(0)
        z = distance * np.cos(theta)
        camera_origin = np.array([x, y, z])
        # camera_origin = sample_point_in_sphere(distance, self.top_only)
        # camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, -np.pi/2, self.shift)
        self.light.light.intensity = intensity
        self.scene.set_pose(self.camera, camera_to_world)
        z_angle = np.random.uniform(0, 2*np.pi)
        z_rotation = np.array(
                    [[np.cos(z_angle), -np.sin(z_angle), 0., 0],
                     [np.sin(z_angle), +np.cos(z_angle), 0.0, 0],
                     [0., 0., 1.0, 0],
                     [0., 0., 0.0, 1]])
        self.scene.set_pose(self.mesh, z_rotation)
        self.scene.set_pose(self.light, camera_to_world)
        image, depth = self.renderer.render(self.scene, flags=self.RGBA)
        image, alpha = split_alpha_channel(image)
        x_min, y_min, x_max, y_max = compute_box_from_mask(alpha, 255)
        image = image[y_min:y_max, x_min:x_max]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = alpha[y_min:y_max, x_min:x_max]
        return image, alpha


class DictionaryView():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        top_only: Boolean. If True images are only take from the top.
        light: List of floats indicating [max_light, min_light]
    """
    def __init__(self, filepath, viewport_size=(128, 128),
                 y_fov=3.14159 / 4., distance=0.30, top_only=False,
                 light=5.0, theta_steps=10, phi_steps=10,):
        self.scene = Scene(bg_color=[0, 0, 0], ambient_light=[0.1255, 0.1255, 0.1255])
        # Bring camera as close to linemod camera
        self.camera = self.scene.add(PerspectiveCamera(
            y_fov, aspectRatio=np.divide(*viewport_size)))
        self.mesh = self.scene.add(Mesh.from_trimesh(
            trimesh.load(filepath), smooth=True))
        self.world_origin = self.mesh.mesh.centroid
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], light))
        self.distance = distance
        # 0.1 values are to avoid gimbal lock
        theta_max = np.pi / 2.0 if top_only else np.pi
        self.thetas = np.linspace(np.pi/2, 0.00, theta_steps)
        # self.thetas = np.linspace(0, np.pi/2, theta_steps) # This theta works
        self.obj_z_rotation = np.linspace(0.00, 2 * np.pi, phi_steps)
        self.renderer = OffscreenRenderer(*viewport_size)
        self.RGBA = RenderFlags.RGBA

    def render(self):
        dictionary_data = []
        for theta_arg, theta in enumerate(self.thetas):
            for phi_arg, z_angle in enumerate(self.obj_z_rotation):
                x = self.distance * np.sin(theta) * np.cos(0)
                y = self.distance * np.sin(theta) * np.sin(0)
                z = self.distance * np.cos(theta)
                matrices = compute_modelview_matrices(
                    np.array([x, y, z]), self.world_origin, roll=-np.pi/2) #x z y works
                camera_to_world, world_to_camera = matrices
                self.scene.set_pose(self.camera, camera_to_world)
                self.scene.set_pose(self.light, camera_to_world)
                z_rotation = np.array(
                    [[np.cos(z_angle), -np.sin(z_angle), 0., 0],
                     [np.sin(z_angle), +np.cos(z_angle), 0.0, 0],
                     [0., 0., 1.0, 0],
                     [0., 0., 0.0, 1]])
                self.scene.set_pose(self.mesh, z_rotation)
                camera_to_world = camera_to_world.flatten()
                world_to_camera = world_to_camera.flatten()
                # viewer.Viewer(self.scene)
                image, depth = self.renderer.render(
                    self.scene, flags=self.RGBA)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image, alpha = split_alpha_channel(image)
                x_min, y_min, x_max, y_max = compute_box_from_depth(depth, 0)
                image = image[y_min:y_max, x_min:x_max]
                image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
                HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8)
                H, S, V = HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2]
                S = (S*1.3).astype(np.uint8)
                # V = (V*1.0).astype(np.uint8)
                H = H[:, :, np.newaxis]
                S = S[:, :, np.newaxis]
                V = V[:, :, np.newaxis]
                HSV_composed = np.concatenate((H, S, V), axis=-1)
                image = cv2.cvtColor(HSV_composed, cv2.COLOR_HSV2RGB)
                # image = np.power(image, 1/1.2).clip(0,255).astype(np.uint8)
                matrices = np.vstack([world_to_camera, camera_to_world])
                sample = {'image': image,
                          'alpha': alpha,
                          'depth': depth, 'matrices': matrices,
                          'bb_syn': [x_min, y_min, x_max, y_max],
                          'world_to_camera': world_to_camera,
                          'mesh_2_world': z_rotation,
                          't_syn': [x, y, z]}
                dictionary_data.append(sample)
        return dictionary_data


def compute_box_from_mask(mask, mask_value):
    """Computes bounding box from mask image.

    # Arguments
        mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.

    # Returns:
        box: List containing box coordinates.
    """
    masked = np.where(mask == mask_value)
    mask_x, mask_y = masked[1], masked[0]
    if mask_x.size <= 0 or mask_y.size <= 0:
        box = [0, 0, 0, 0]
    else:
        x_min, y_min = np.min(mask_x), np.min(mask_y)
        x_max, y_max = np.max(mask_x), np.max(mask_y)
        box = [x_min, y_min, x_max, y_max]
    return box


def compute_box_from_depth(depth, bg_value):
    """Computes bounding box from mask image.

    # Arguments
        mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.

    # Returns:
        box: List containing box coordinates.
    """
    masked = np.where(depth > bg_value)
    mask_x, mask_y = masked[1], masked[0]
    if mask_x.size <= 0 or mask_y.size <= 0:
        box = [0, 0, 0, 0]
    else:
        x_min, y_min = np.min(mask_x), np.min(mask_y)
        x_max, y_max = np.max(mask_x), np.max(mask_y)
        box = [x_min, y_min, x_max, y_max]
    return box
