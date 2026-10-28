
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GameCNN(keras.Model):
    def __init__(self, filters=64):
        super().__init__()

        # Shared feature extractor
        self.conv1 = layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )

        # ACTOR: one raw score (logit) per board position
        self.policy_head = layers.Conv2D(
            1, kernel_size=1, padding="same"
        )

        # CRITIC: one value estimate for the whole board
        self.pool = layers.GlobalAveragePooling2D()
        self.value_dense = layers.Dense(64, activation="relu")
        self.value_head = layers.Dense(1, activation="tanh")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Actor output: (batch, rows, columns)
        policy_logits = tf.squeeze(
            self.policy_head(x), axis=-1
        )

        # Critic output: (batch,)
        value = self.pool(x)
        value = self.value_dense(value)
        value = tf.squeeze(self.value_head(value), axis=-1)

        return policy_logits, value
