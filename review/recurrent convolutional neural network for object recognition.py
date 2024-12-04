import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

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

# 모델 컴파일
rcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# 훈련
history = rcnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
