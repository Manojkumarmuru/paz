from paz.abstract import Processor
import numpy as np
from operator import itemgetter


class MakeDictionary(Processor):
    def __init__(self, encoder, renderer):
        super(MakeDictionary, self).__init__()
        self.latent_dimension = encoder.encoder.output_shape[1]
        self.encoder = encoder
        self.renderer = renderer

    def call(self):
        data = self.renderer.render()
        latent_vectors = np.zeros((len(data), self.latent_dimension))
        for sample_arg, sample in enumerate(data):
            image = sample['image']
            latent_vectors[sample_arg] = self.encoder(image)
        data.append(latent_vectors)
        return data


class MeasureSimilarity(Processor):
    def __init__(self, dictionary, measure):
        super(MeasureSimilarity, self).__init__()
        self.dictionary = dictionary
        self.measure = measure

    def call(self, latent_vector):
        latent_vectors = self.dictionary[-1]
        measurements = self.measure(latent_vectors, latent_vector)
        k = 10
        top_k = list(np.argsort(measurements, axis=0)[-k:, 0])
        closest_images = [self.dictionary[top] for top in top_k]
        return latent_vector, closest_images
