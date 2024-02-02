from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import EncoderPredictor, DecoderPredictor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr

from processors import MeasureSimilarity
from processors import MakeDictionary
import cv2
import numpy as np


class ImplicitRotationPredictor(Processor):
    def __init__(self, encoder, decoder, measure, renderer):
        super(ImplicitRotationPredictor, self).__init__()
        self.show_decoded_image = pr.ShowImage('decoded_image', wait=False)
        self.show_closest_image1 = pr.ShowImage('closest_image1', wait=False)
        self.show_closest_image2 = pr.ShowImage('closest_image2', wait=False)
        self.show_closest_image3 = pr.ShowImage('closest_image3', wait=False)
        self.show_closest_image4 = pr.ShowImage('closest_image4', wait=False)
        self.show_closest_image5 = pr.ShowImage('closest_image5', wait=False)
        self.show_closest_image6 = pr.ShowImage('closest_image6', wait=False)
        self.show_closest_image7 = pr.ShowImage('closest_image7', wait=False)
        self.show_closest_image8 = pr.ShowImage('closest_image8', wait=False)
        self.show_closest_image9 = pr.ShowImage('closest_image9', wait=False)
        self.show_closest_image10 = pr.ShowImage('closest_image10', wait=False)
        self.encoder = EncoderPredictor(encoder)
        self.dictionary = MakeDictionary(self.encoder, renderer)()
        self.encoder.add(pr.ExpandDims(0))
        self.encoder.add(MeasureSimilarity(self.dictionary, measure))
        self.decoder = DecoderPredictor(decoder)
        outputs = ['image', 'latent_vector', 'latent_image', 'decoded_image',
                   't_real_z']
        self.wrap = pr.WrapOutput(outputs)

    def call(self, image, t_syn, f_syn, f_real, raw_im_dims):
        latent_vector, closest_images = self.encoder(image)
        latent_vector = latent_vector
        self.show_closest_image1(closest_images[0][0])
        self.show_closest_image2(closest_images[1][0])
        self.show_closest_image3(closest_images[2][0])
        self.show_closest_image4(closest_images[3][0])
        self.show_closest_image5(closest_images[4][0])
        self.show_closest_image6(closest_images[5][0])
        self.show_closest_image7(closest_images[6][0])
        self.show_closest_image8(closest_images[7][0])
        self.show_closest_image9(closest_images[8][0])
        self.show_closest_image10(closest_images[9][0])
        decoded_image = self.decoder(latent_vector[0])
        t_real_zs = self.compute_t_real_z(image, t_syn, f_syn, f_real,
                                          closest_images, raw_im_dims)
        self.show_decoded_image(decoded_image)
        return self.wrap(image, latent_vector, closest_images,
                         decoded_image, t_real_zs)

    def compute_t_real_z(self, image, t_syn, f_syn, f_real,
                         closest_images, raw_im_dims):
        x_min, y_min, x_max, y_max = raw_im_dims
        real_diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        t_real_zs = []
        for i in range(len(closest_images)):
            x_min, y_min, x_max, y_max = closest_images[i][1]
            syn_diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            t_real_z = t_syn * (syn_diag/real_diag) * (f_real/f_syn)
            t_real_zs.append(t_real_z)
        return t_real_zs

class DomainRandomizationProcessor(Processor):
    def __init__(self, renderer, image_paths, num_occlusions, split=pr.TRAIN):
        super(DomainRandomizationProcessor, self).__init__()
        self.copy = pr.Copy()
        self.render = pr.Render(renderer)
        self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        preprocessors = [pr.ConvertColorSpace(pr.RGB2BGR), pr.NormalizeImage()]
        self.preprocess = SequentialProcessor(preprocessors)
        self.split = split

    def call(self):
        input_image, alpha_mask = self.render()
        label_image = self.copy(input_image)
        if self.split == pr.TRAIN:
            input_image = self.augment(input_image, alpha_mask)
        input_image = self.preprocess(input_image)
        label_image = self.preprocess(label_image)
        input_image = cv2.resize(input_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        label_image = cv2.resize(label_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        return input_image, label_image


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


class DomainRandomization(SequentialProcessor):
    def __init__(self, renderer, shape, image_paths,
                 num_occlusions, split=pr.TRAIN):
        super(DomainRandomization, self).__init__()
        self.add(DomainRandomizationProcessor(
            renderer, image_paths, num_occlusions, split))
        self.add(pr.SequenceWrapper(
            {0: {'input_image': [128, 128, 3]}},
            {1: {'label_image': [128, 128, 3]}}))
