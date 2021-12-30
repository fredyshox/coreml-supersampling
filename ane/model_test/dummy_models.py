from re import L
import tensorflow as tf


class CumsumModel(tf.keras.models.Model):
    def __init__(self, rec_model, axis=3, reverse=False):
        name = f"CumsumModel_Axis{axis}{'_Reverse' if reverse else ''}"
        super().__init__(name=name)

        self.reverse = reverse 
        self.axis = axis
        self.reconstruction = rec_model

    def call(self, x):
        h = tf.cumsum(x, axis=self.axis, reverse=self.reverse)
        return self.reconstruction(h)


class CustomCumsumModelAxis3(tf.keras.models.Model):
    def __init__(self, rec_model, shape):
        name = f"CustomCumsumModel_Axis3_Reverse"
        super().__init__(name=name)

        self.n_channels = shape[-1]
        self.plane_shape = shape[:-1]
        self.reconstruction = rec_model

    def call(self, x):
        tensors = []
        hh = tf.zeros(self.plane_shape)
        for i in range(self.n_channels):
            hh = tf.add(hh, x[:, :, :, i])
            tensors.append(hh)
        
        result = tf.stack(tensors, axis=3)

        return self.reconstruction(result)


class ArithmenticsWithBroadcastingModel(tf.keras.models.Model):
    def __init__(self, rec_model, shape, same_shape=False, scalar=False, broadcasting=False):
        suffix = ""
        suffix = suffix + ("_SameShape" if same_shape else "")
        suffix = suffix + ("_Scalar" if scalar else "")
        suffix = suffix + ("_Broadcasting" if broadcasting else "")
        name = f"ArithmenticsWithBroadcastingModel{suffix}"
        super().__init__(name=name)

        self.reconstruction = rec_model
        self.same_shape = same_shape
        self.scalar = scalar
        self.broadcasting = broadcasting
        if same_shape:
            self.add_tensor_same_shape = tf.random.uniform(shape)
            self.sub_tensor_same_shape = tf.random.uniform(shape)
            self.mul_tensor_same_shape = tf.random.uniform(shape)
        if broadcasting:
            self.add_tensor_with_broadcasting = tf.random.uniform(shape[2:])
            self.sub_tensor_with_broadcasting = tf.random.uniform(shape[2:])
            self.mul_tensor_with_broadcasting = tf.random.uniform(shape[2:])
        if scalar:
            self.add_element = tf.random.uniform(())
            self.sub_element = tf.random.uniform(())
            self.mul_element = tf.random.uniform(())

    def call(self, x):
        h = x 
        if self.same_shape:
            h = tf.add(h ,self.add_tensor_same_shape)
            h = tf.subtract(h, self.sub_tensor_same_shape)
            h = tf.multiply(h, self.mul_tensor_same_shape)
        if self.scalar:
            h = tf.add(h, self.add_element)
            h = tf.subtract(h, self.sub_element)
            h = tf.multiply(h, self.mul_element)
        if self.broadcasting:
            h = tf.add(h, self.add_tensor_with_broadcasting)
            h = tf.subtract(h, self.sub_tensor_with_broadcasting)
            h = tf.multiply(h, self.mul_tensor_with_broadcasting)
        return self.reconstruction(h)


class MaxMinModel(tf.keras.models.Model):
    def __init__(self, rec_model):
        super().__init__(name="MaxMinModel")
        self.reconstruction = rec_model

    def call(self, x):
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        # do something about it
        h = tf.subtract(x, x_min)
        h = tf.multiply(h, 1.0/(x_max - x_min))
        return self.reconstruction(h)


class FloorModel(tf.keras.models.Model):
    def __init__(self, rec_model):
        super().__init__(name="FloorModel")
        self.reconstruction = rec_model

    def call(self, x):
        h = tf.floor(x)
        return self.reconstruction(h)


class GatherModel(tf.keras.models.Model):
    def __init__(self, rec_model, shape, axis=1):
        super().__init__(name=f"GatherModel_Axis{axis}")
        self.reconstruction = rec_model
        self.axis = axis
        self.gather_range = tf.range(shape[axis], delta=2)
    
    def call(self, x):
        h = tf.gather(x, self.gather_range, axis=self.axis)
        h = tf.concat([h, h], axis=self.axis)
        return self.reconstruction(h)


class SplitModel(tf.keras.models.Model):
    def __init__(self, rec_model, num_splits=4, axis=1):
        super().__init__(name=f"SplitModel_Splits{num_splits}_Axis{axis}")
        self.reconstruction = rec_model
        self.num_splits = num_splits
        self.axis = axis

    def call(self, x):
        splits = tf.split(x, self.num_splits, axis=self.axis)
        h = splits[0]
        h = tf.concat([h] * self.num_splits, axis=self.axis)
        return self.reconstruction(h)
