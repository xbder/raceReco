import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

print(tf.__version__)

# configs
DATA_DIR = 'F:/workspace/dataset/UTKFace'
# ! ls /home/jackon/datasets/UTKFace/UTKFace | wc -l

TRAIN_TEST_SPLIT = 0.7
IMAGE_HEIGHT, IMAGE_WIDTH = 198, 198
IMAGE_CHANNELS = 3
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = {g: i for i, g in ID_GENDER_MAP.items()}
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = {r: i for i, r in ID_RACE_MAP.items()}

print(ID_GENDER_MAP, GENDER_ID_MAP, ID_RACE_MAP, RACE_ID_MAP)

'''
    通过文件名解析出年龄、性别、人种
'''
def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), ID_GENDER_MAP[int(gender)], ID_RACE_MAP[int(race)]
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None, None

# create a pandas data frame of images, age, gender and race
files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))

df_origin = pd.DataFrame(attributes)
df_origin['file'] = files
df_origin.columns = ['age', 'gender', 'race', 'file']
df_origin = df_origin.dropna()
df_origin.head()

df_origin.describe()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df_origin, x='gender', y='age', ax=ax1)
sns.boxplot(data=df_origin, x='race', y='age', ax=ax2)

plt.figure(figsize=(15, 6))
sns.boxplot(data=df_origin, x='gender', y='age', hue='race')

df_origin.groupby(by=['race', 'gender'])['age'].count().plot(kind='bar')

df_origin['age'].hist()

df_origin['age'].describe()

df = df_origin.copy()
df = df[(df['age'] > 10) & (df['age'] < 65)]
df.describe()

df.head()

## data processing
p = np.random.permutation(len(df))
train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])
df['race_id'] = df['race'].map(lambda race: RACE_ID_MAP[race])

max_age = df['age'].max()
print('train count: %s, valid count: %s, test count: %s, max age: %s' % (
    len(train_idx), len(valid_idx), len(test_idx), max_age))

## 生成器
from tensorflow.keras.utils import to_categorical
from PIL import Image

IM_WIDTH, IM_HEIGHT = 198, 198
def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IMAGE_HEIGHT, IMAGE_HEIGHT))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            races.append(to_categorical(race, len(RACE_ID_MAP)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break

# model training
def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x

W, H, C = 100, 80, 3
N_LABELS = 10
D = 2
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

input_layer = tf.keras.Input(shape=(W, H, C))
x = layers.Conv2D(32, 3, activation='relu')(input_layer)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)

num_res_net_blocks = 10
for i in range(num_res_net_blocks):
  x = res_net_block(x, 64, 3)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
bottleneck = layers.Dropout(0.5)(x)

# for age calculation
age_x = Dense(128, activation='relu')(bottleneck)
age_output = Dense(1, activation='sigmoid', name='age_output')(age_x)

# for race prediction
race_x = Dense(128, activation='relu')(bottleneck)
race_output = Dense(len(RACE_ID_MAP), activation='softmax', name='race_output')(race_x)

# for gender prediction
gender_x = Dense(128, activation='relu')(bottleneck)
gender_output = Dense(len(GENDER_ID_MAP), activation='softmax', name='gender_output')(gender_x)

model = models.Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])
model.compile(optimizer='rmsprop',
              loss={
                  'age_output': 'mse',
                  'race_output': 'categorical_crossentropy',
                  'gender_output': 'categorical_crossentropy'},
              loss_weights={
                  'age_output': 2.,
                  'race_output': 1.5,
                  'gender_output': 1.},
              metrics={
                  'age_output': 'mae',
                  'race_output': 'accuracy',
                  'gender_output': 'accuracy'})
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=10,
#                     callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)

print('\n'.join(history.history.keys()))


# def plot_train_history(history):
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     axes[0].plot(history.history['race_output_accuracy'], label='Race Train accuracy')
#     axes[0].plot(history.history['val_race_output_accuracy'], label='Race Val accuracy')
#     axes[0].set_xlabel('Epochs')
#     axes[0].legend()
#
#     axes[1].plot(history.history['gender_output_accuracy'], label='Gender Train accuracy')
#     axes[1].plot(history.history['val_gender_output_accuracy'], label='Gener Val accuracy')
#     axes[1].set_xlabel('Epochs')
#     axes[1].legend()
#
#     axes[2].plot(history.history['age_output_mae'], label='Age Train MAE')
#     axes[2].plot(history.history['val_age_output_mae'], label='Age Val MAE')
#     axes[2].set_xlabel('Epochs')
#     axes[2].legend()
#
#     axes[3].plot(history.history['loss'], label='Training loss')
#     axes[3].plot(history.history['val_loss'], label='Validation loss')
#     axes[3].set_xlabel('Epochs')
#     axes[3].legend()
#
#
# plot_train_history(history)

test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//128)))

test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
x_test, (age_true, race_true, gender_true)= next(test_gen)
age_pred, race_pred, gender_pred = model.predict_on_batch(x_test)

race_pred, gender_pred = tf.math.argmax(race_pred, axis=1), tf.math.argmax(gender_pred, axis=1)
race_true, gender_true = tf.math.argmax(race_true, axis=1), tf.math.argmax(gender_true, axis=1)
age_true = age_true * max_age
age_pred = age_pred * max_age

import math
n = 30
random_indices = np.random.permutation(n)
n_cols = 5
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(x_test[img_idx])
    ax.set_title('(pred)a:{}, g:{}, r:{}'.format(
        int(age_pred[img_idx].numpy()),
        ID_GENDER_MAP[gender_pred[img_idx].numpy()],
        ID_RACE_MAP[race_pred[img_idx].numpy()]))
    ax.set_xlabel('(true)a:{}, g:{}, r:{}'.format(
        int(age_true[img_idx]), ID_GENDER_MAP[gender_true[img_idx].numpy()], ID_RACE_MAP[race_true[img_idx].numpy()]))
    ax.set_xticks([])
    ax.set_yticks([])


from sklearn.metrics import classification_report
print("Classification report for race")
print(classification_report(race_true, race_pred, labels=list(ID_RACE_MAP.keys())))

print("\nClassification report for gender")
print(classification_report(gender_true, gender_pred, labels=list(ID_GENDER_MAP.keys())))
