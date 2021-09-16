import tensorflow as tf
from tensorflow.keras.layers import Layer
from backend import find_max_location


class SegmentationDilation(Layer):
    def __init__(self, filter_size=21):
        super(SegmentationDilation, self).__init__()
        self.filter_size = filter_size
        self.kernel = tf.ones((self.filter_size, self.filter_size, 1)) / float(
            self.filter_size ** 2)  # move it to call or init

    def call(self, inputs):
        shape = inputs.shape

        scoremap_softmax = tf.nn.softmax(inputs)
        scoremap_fg = tf.reduce_max(scoremap_softmax[:, :, :, 1:],
                                    -1)  # B, H, W
        detmap_fg = tf.round(scoremap_fg)  # B, H, W

        max_loc = find_max_location(scoremap_fg)

        objectmap_list = list()
        kernel_dil = tf.ones((self.filter_size, self.filter_size, 1)) / float(
            self.filter_size * self.filter_size)
        if shape[0] is None:
            shape[0] = 1
        for i in range(shape[0]):
            sparse_ind = tf.reshape(max_loc[i, :], [1, 2])
            objectmap = tf.compat.v1.sparse_to_dense(sparse_ind, [shape[1],
                                                                  shape[2]],
                                                     1.0)

            num_passes = max(shape[1], shape[2]) // (self.filter_size // 2)
            for pass_count in range(num_passes):
                objectmap = tf.reshape(objectmap, [1, shape[1], shape[2], 1])
                objectmap_dil = tf.compat.v1.nn.dilation2d(objectmap,
                                                           kernel_dil,
                                                           [1, 1, 1, 1],
                                                           [1, 1, 1, 1], 'SAME')

                objectmap_dil = tf.reshape(objectmap_dil, [shape[1], shape[2]])
                objectmap = tf.round(
                    tf.multiply(detmap_fg[i, :, :], objectmap_dil))

            objectmap = tf.reshape(objectmap, [shape[1], shape[2], 1])
            objectmap_list.append(objectmap)

        objectmap = tf.stack(objectmap_list)

        return objectmap
