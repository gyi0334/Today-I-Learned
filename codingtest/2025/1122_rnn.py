import tensorflow as tf
tf.random.set_seed(1)
rnn_layer = tf.keras.layers.SimpleRNN(units=2, use_bias=True, return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layer.weights
print('W_xh 크기: ',w_xh.shape)
print('W_oo 크기: ',w_oo.shape)
print('b_h 크기: ',b_h.shape)

x_seq = tf.convert_to_tensor([[1.0]*5, [2.0]*5, [3.0]*5], dtype=tf.float32)
# simplernn 의 출력
output = rnn_layer(tf.reshape(x_seq, shape=(1,3,5)))
# 수동으로 출력 계산하기
out_man=[]
for t in range(len(x_seq)):  # len(x_seq) = 3
    xt = tf.reshape(x_seq[t], (1,5))
    print('time step {} => '.format(t))
    print('  input            : ',xt.numpy())

    ht = tf.matmul(xt, w_xh) + b_h
    print('  hidden           : ',ht.numpy())

    if t>0:
        prev_o = out_man[t-1]