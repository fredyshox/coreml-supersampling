from tensorflow import is_tensor
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

class PerceptualFPVGG16(Model):
    """
    VGG16 wrapper with input in floating-point format (0...1) with activation outputs
    """
    def __init__(self, weights, input_shape, output_layer_names, loc, scale, trainable=False):
        super().__init__()
        
        vgg_model = VGG16(include_top=False, weights=weights, input_shape=input_shape)
        self.loc = loc # mean
        self.scale = scale # stddev
        self.custom_vgg_model = Model(vgg_model.input, [vgg_model.get_layer(layer).output for layer in output_layer_names])
        self.custom_vgg_model.trainable = trainable
        self.trainable = trainable
    
    def call(self, x):
        """
        x is expected in float image format, that is 0...1 range of values
        """
        scaled_x = x * 255.0
        preprocessed_x = preprocess_input(scaled_x)
        output = self.custom_vgg_model(preprocessed_x)
        if is_tensor(output):
            standardized_output = (output - self.loc) / self.scale
        else: # must be list then (multi output)
            standardized_output = [(activation - self.loc) / self.scale for activation in output]

        return standardized_output