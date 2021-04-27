import argparse
import tensorflow as tf

import efficientnet_builder
from efficientdet_building_blocks import ResampleFeatureMap
from config import default_configuration

# Mock input image.
mock_input_image = tf.random.uniform((1, 224, 224, 3), dtype=tf.dtypes.float32, seed=1)


class EfficientDet(tf.keras.Model):
    """
    EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """

    def __init__(self, config, name=""):
        """Initialize model.
        # Arguments
            config: Configuration of the EfficientDet model.
            name: A string of layer name.
        """
        super().__init__(name=name)

        self.config = config
        self.backbone = efficientnet_builder.build_backbone(config)
        self.resample_layers = []
        for level in range(6, config["max_level"] + 1):
            self.resample_layers.append(ResampleFeatureMap(
                feat_level=(level - config["min_level"]),
                target_num_channels=config["fpn_num_filters"],
                apply_bn=config["apply_bn_for_resampling"],
                conv_after_downsample=config["conv_after_downsample"],
                data_format=config["data_format"],
                name='resample_p%d' % level,
            ))

    def call(self, images, training=False):
        """Build EfficientDet model.
        # Arguments
            images: Tensor, indicating the image input to the architecture.
            training: Bool, whether EfficientDet architecture is trained.
        """

        # Efficientnet backbone features
        all_feats = self.backbone(images)

        feats = all_feats[config["min_level"] - 1: config["max_level"] + 1]

        # Build additional input features that are not from backbone.
        for resample_layer in self.resample_layers:
            feats.append(resample_layer(feats[-1], training, None))

        # BiFPN layers
        # TODO: Implement BiFPN

        # Classification head
        # TODO: Implement classification head
        class_outputs = None

        # Box regression head
        # TODO: Implement box regression head
        box_outputs = None

        return class_outputs, box_outputs


if __name__ == "__main__":

    description = "Build EfficientDet model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-m",
        "--model_name",
        default="EfficientDetD0",
        type=str,
        help="EfficientDet model name",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--backbone_name",
        default="EfficientNetB0",
        type=str,
        help="EfficientNet backbone name",
        required=False,
    )
    parser.add_argument(
        "-bw",
        "--backbone_weight",
        default="imagenet",
        type=str,
        help="EfficientNet backbone weight",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--act_type",
        default="swish",
        type=str,
        help="Activation function",
        required=False,
    )
    parser.add_argument(
        "--data_format",
        default="channels_last",
        type=str,
        help="Channels position in the data. "
        "Two options: channels_first and channels_last for "
        "(channels, image_height, image_width) or "
        "(image_height, image_width, channels), respectively",
        required=False,
    )
    parser.add_argument(
        "--min_level",
        default=3,
        type=int,
        help="EfficientNet feature minimum level. "
        "Level decides the activation map size, "
        "eg: For an input image of 640 x 640, "
        "the activation map resolution at level 3 is "
        "(640 / (2 ^ 3)) x (640 / (2 ^ 3))",
        required=False,
    )
    parser.add_argument(
        "--max_level",
        default=7,
        type=int,
        help="EfficientNet feature maximum level. "
        "Level decides the activation map size,"
        " eg: For an input image of 640 x 640, "
        "the activation map resolution at level 3 is"
        " (640 / (2 ^ 3)) x (640 / (2 ^ 3))",
        required=False,
    )
    args = parser.parse_args()
    config = vars(args)
    config.update(default_configuration)
    print(config)
    # TODO: Add parsed user-inputs to the config and update the config
    efficientdet = EfficientDet(config=config)
    class_outputs, box_outputs = efficientdet(mock_input_image, False)
