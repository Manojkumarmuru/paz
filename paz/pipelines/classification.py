from ..abstract import SequentialProcessor
from .. import processors as pr
from . import PreprocessImage
from ..models.classification import MiniXception, VVAD_LRS3_LSTM, CNN2Plus1D, CNN2Plus1D_Filters
from ..datasets import get_class_names
from .keypoints import MinimalHandPoseEstimation


# neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]


class MiniXceptionFER(SequentialProcessor):
    """Mini Xception pipeline for classifying emotions from RGB faces.

    # Example
        ``` python
        from paz.pipelines import MiniXceptionFER

        classify = MiniXceptionFER()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``class_names`` and ``scores``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)

    """
    def __init__(self):
        super(MiniXceptionFER, self).__init__()
        self.classifier = MiniXception((48, 48, 1), 7, weights='FER')
        self.class_names = get_class_names('FER')

        preprocess = PreprocessImage(self.classifier.input_shape[1:3], None)
        preprocess.insert(0, pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.ExpandDims(0))
        preprocess.add(pr.ExpandDims(-1))
        self.add(pr.Predict(self.classifier, preprocess))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.ToClassName(self.class_names), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))


class ClassifyHandClosure(SequentialProcessor):
    """Pipeline to classify minimal hand closure status.

    # Example
        ``` python
        from paz.pipelines import ClassifyHandClosure

        classify = ClassifyHandClosure()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs an image with class
        status drawn on it.
    """
    def __init__(self, draw=True, right_hand=False):
        super(ClassifyHandClosure, self).__init__()
        self.add(MinimalHandPoseEstimation(draw, right_hand))
        self.add(pr.UnpackDictionary(['image', 'relative_angles']))
        self.add(pr.ControlMap(pr.IsHandOpen(), [1], [1]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage('OPEN', 'CLOSE'),
                               [1], [1]))
        if draw:
            self.add(pr.ControlMap(pr.DrawText(), [0, 1], [0], {1: 1}))
        self.add(pr.WrapOutput(['image', 'status']))


class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for classifying speaking and not speaking from RGB faces.
    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (frames, height, width, channels)
            e.g. (38, 96, 96, 3).
        update_rate: Integer. How many frames are between the prediction (computational expansive (low update rate) vs
            high latency (high update rate))
        average: Integer. How many predictions are averaged
    """
    def __init__(self, input_size=(38, 96, 96, 3), update_rate=38, average=2):
        super(ClassifyVVAD, self).__init__()
        # self.classifier = VVAD_LRS3_LSTM(weights='VVAD-LRS3')
        # self.classifier = CNN2Plus1D(weights='CNN2Plus1D')
        self.classifier = CNN2Plus1D_Filters(weights='CNN2Plus1D')
        self.class_names = get_class_names('VVAD-LRS3')

        preprocess = PreprocessImage(input_size[1:3], pr.RGB_NORMAL)
        preprocess.add(pr.BufferImages( input_size, play_rate=update_rate))
        self.add(pr.Predict(self.classifier, preprocess))
        self.add(pr.AveragePredictions(average))
        self.add(pr.CopyDomain([0], [0]))
        self.add(pr.ControlMap(pr.NoneConverter(), [0], [0]))
        self.add(pr.ControlMap(pr.FloatToBoolean(), [0], [0]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage(true_message=self.class_names[0], false_message=self.class_names[1]), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))
