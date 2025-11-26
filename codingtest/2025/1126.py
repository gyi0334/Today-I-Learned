import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

# 텍스트 읽고 처리하기
with open('1268-0.txt', 'r', encoding='UTF8') as fp:
    text = fp.read()
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)
print('total length : ', len(text))
print('고유한 문자 : ', len(char_set))

chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('인코딩된 텍스트 크기 : ', text_encoded.shape)
print(text[:15], '            == 인코딩 ==> ', text_encoded[:15])
print(text_encoded[15:21], '            == 디코딩 ==> ', ''.join(char_array[text_encoded[15:21]]))

ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)
for ex in ds_text_encoded.take(5):
    print('{} -> {}'.format(ex.numpy(), char_array[ex.numpy()]))

seq_length = 40
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)
# x & y 를 나누기 위함 함수를 정의합니다.
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq
ds_sequences = ds_chunks.map(split_input_target)

for example in ds_sequences.take(2):
    print('입력 (x): ', repr(''.join(char_array[example[0].numpy()])))
    print('출력 (y): ', repr(''.join(char_array[example[1].numpy()])))
    print()

BATCH_SIZE=64
BUFFER_SIZE=10000
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim),
                                 tf.keras.layers.LSTM(rnn_units, return_sequences=True),
                                 tf.keras.layers.Dense(vocab_size)])
    return model

## 매개변수 설정
charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512
tf.random.set_seed(1)
model = build_model(vocab_size=charset_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(ds, epochs=20)