import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras import mixed_precision

size = (120, 120)


def build_model(size=size):
    input_layer = Input(shape=(size[0], size[1], 3))
    vgg = VGG16(include_top=False)(input_layer)

    # * 1 for classification
    f1 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation="relu")(f1)
    class2 = Dense(6, activation="softmax")(class1)
    # class2 = tf.cast(class2, tf.float16)

    # * 4 for bounding box
    f2 = tf.keras.layers.GlobalAveragePooling2D()(vgg)
    reggress1 = Dense(2048, activation="relu")(f2)
    reggress2 = Dense(4, activation="sigmoid")(reggress1)
    # reggress2 = tf.cast(reggress2, tf.float16)

    face_tracker = Model(inputs=input_layer, outputs=[class2, reggress2])

    return face_tracker



def localization_loss(y_true, y_pred):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - y_pred[:, :2]))

    try:
        h_true = y_true[:, 3] - y_true[:, 1]
        w_true = y_true[:, 2] - y_true[:, 0]

        h_pred = y_pred[:, 3] - y_pred[:, 1]
        w_pred = y_pred[:, 2] - y_pred[:, 0]
    except Exception as e:
        print(e)
        print(y_true)
        print(y_pred)
        raise e

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    # delta_size = tf.reduce_sum(tf.square(tf.sqrt(w_true) - tf.sqrt(w_pred)) + tf.square(tf.sqrt(h_true) - tf.sqrt(h_pred)))

    return delta_coord + 0.5 * delta_size



class FaceTracker(Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.coord_track = []

    def compile(self, optimizer, class_loss, regression_loss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.class_loss = class_loss
        self.regression_loss = regression_loss

    def train_step(self, batch, **kwargs):
        x, y = batch

        with tf.GradientTape() as tape:
            # * predict
            classes, coords = self.model(x, training=True)
            # self.coord_track.append(coords)
            # * calculate loss
            batch_class_loss = self.class_loss(y[0], classes)
            batch_regression_loss = self.regression_loss(y[1], coords)

            # * total loss
            total_loss = 2 * batch_regression_loss + batch_class_loss

            # * get gradients
            grad = tape.gradient(total_loss, self.model.trainable_variables)

        # * update weights
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return {
            "loss": total_loss,
            "class_loss": batch_class_loss,
            "regression_loss": batch_regression_loss,
        }

    def test_step(self, batch, **kwargs):
        x, y = batch

        classes, coords = self.model(x, training=False)
        batch_class_loss = self.class_loss(y[0], classes)
        batch_regression_loss = self.regression_loss(y[1], coords)

        total_loss = batch_regression_loss + batch_class_loss
        return {
            "loss": total_loss,
            "class_loss": batch_class_loss,
            "regression_loss": batch_regression_loss,
        }

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)
