import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.tail())

target = df.pop('sentiment')

ds_raw = tf.data.Dataset.from_tensor_slices((df['review'].astype(str).to_numpy(),
                                             target.astype('int32').to_numpy()))

for text, label in ds_raw.take(3):
    short = tf.strings.substr(text, 0, 50)   # 0부터 50글자 잘라내기
    tf.print(short, label)

tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# STEP 02. 고유 토큰(단어) 찾기
from collections import Counter
tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy())
    token_counts.update(tokens)
print('어휘 사전 크기 : ', len(token_counts))

