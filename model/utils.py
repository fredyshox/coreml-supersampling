import tensorflow as tf


def tf_minor_version_geq(x):
    version = tf.__version__.split(".")
    return int(version[1]) >= x


def ELEM_IF(condition, element):
    return [element] if condition else []