import tensorflow as tf
from tensorflow.keras import layers, models

# RCL (Recurrent Convolutional Layer) 정의
class RCL(layers.Layer):
    def __init__(self, filters, kernel_size, num_iterations):
        super(RCL, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.recurrent_conv = layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.num_iterations = num_iterations

    def call(self, inputs):
        x = self.conv(inputs)
        for _ in range(self.num_iterations):
            x = self.recurrent_conv(x) + x
        return x

# RCNN 모델 정의
def build_rcnn(input_shape=(32, 32, 3), num_classes=10, num_iterations=3):
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: Standard Conv Layer
    x = layers.Conv2D(96, (5, 5), activation='relu', padding="same")(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    
    # Layer 2-5: RCL + MaxPooling
    for filters in [96, 96, 96, 96]:
        x = RCL(filters, kernel_size=(3, 3), num_iterations=num_iterations)(x)
        if filters != 96:  # MaxPooling after certain RCL layers
            x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Global MaxPooling + Fully Connected Layer
    x = layers.GlobalMaxPooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# 모델 생성
rcnn_model = build_rcnn()
rcnn_model.summary()
