import tensorflow as tf

class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2,3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1,2)), trainable=False)

m = MyModule()
print('all parameter : ',[v.shape for v in m.variables])
print('trainable parameter : ', [v.shape for v in m.trainable_variables])


@tf.function
def f(x):
    w = tf.Variable([1,2,3])

print(f([1]))