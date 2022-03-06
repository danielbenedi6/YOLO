import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential, Model


class YoLo:
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C
        self.model = Sequential()

    def __conv_block__(self, filters: int, kernel_size: int, strid: int, apply_batchnorm: bool = True):
        block = Sequential()
        block.add(layers.Conv2D(filters, kernel_size, strides=strid, padding='same'))
        if apply_batchnorm:
            block.add(layers.BatchNormalization())
        block.add(layers.LeakyReLU(0.1))
        return block

    def make(self, input_size=(448, 448, 3)) -> Model:
        input_block = layers.Input(input_size)
        output_block = Sequential()
        output_block.add(layers.Flatten())
        output_block.add(layers.Dense(4096))
        output_block.add(layers.Dropout(0.5))
        output_block.add(layers.LeakyReLU(0.1))
        output_block.add(layers.Dense(self.S * self.S * (self.C + self.B * 5)))
        output_block.add(layers.Reshape((self.S, self.S, self.C + self.B * 5)))

        shape = [
            [(7, 64, 2)],
            [(3, 192, 1)],
            [(1, 128, 1), (3, 256, 1), (1, 256, 1), (3, 512, 1)],
            [(1, 256, 1), (3, 512, 1)] * 4 + [(1, 512, 1), (3, 1024, 1)],
            [(1, 512, 1), (3, 1024, 1)] * 2 + [(3, 1024, 1), (3, 1024, 2)] + [(3, 1024, 1)] * 2
        ]

        tmp = input_block

        for i in range(len(shape)):
            for layer in shape[i]:
                block = self.__conv_block__(kernel_size=layer[0], filters=layer[1], strid=layer[2])
                tmp = block(tmp)
            if i < len(shape) - 1:
                tmp = layers.MaxPooling2D(pool_size=(2, 2))(tmp)



        self.model = Model(input=input_block, outputs=output_block(tmp))
        return self.model
