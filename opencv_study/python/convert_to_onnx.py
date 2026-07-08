import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("mnistcnn.keras")

spec = (
    tf.TensorSpec((None, 28, 28, 1), tf.float32, name="data"),
)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="mnistcnn.onnx"
)

print("ONNX 변환 완료: mnistcnn.onnx")