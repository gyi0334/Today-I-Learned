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

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df['review'].astype(str).to_numpy(),
     target.astype('int32').to_numpy())
)

for text, label in ds_raw.take(3):
    short = tf.strings.substr(text, 0, 50)   # 0부터 50글자 잘라내기
    tf.print(short, label)

