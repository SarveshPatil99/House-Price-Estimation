# Import libraries

import os
from pathlib import Path
import pandas as pd
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm.notebook import tqdm

# Create folders for datasets
Path('New York Data').mkdir(exist_ok = True)
Path('California Data').mkdir(exist_ok = True)
os.system('unzip -q ny_data.zip -d "New York Data/"')
os.system('unzip -q cali_data.zip -d "California Data/"')

# New York Dataset

df = pd.read_pickle('New York Data/df.pkl')
df['image_path'] = 'New York Data/processed_images/'+df['zpid']+'.png'

X_train, X_test, Y_train, Y_test = train_test_split(df,df['price'], test_size = 0.3, random_state = 0)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 0)

max_beds = X_train['beds'].max()
max_baths = X_train['baths'].max()
max_area = X_train['area'].max()
max_price = Y_train.max()

X_train['beds'] = X_train['beds'] / max_beds
X_train['baths'] = X_train['baths'] / max_baths
X_train['area'] = X_train['area'] / max_area

X_val['beds'] = X_val['beds'] / max_beds
X_val['baths'] = X_val['baths'] / max_baths
X_val['area'] = X_val['area'] / max_area

X_test['beds'] = X_test['beds'] / max_beds
X_test['baths'] = X_test['baths'] / max_baths
X_test['area'] = X_test['area'] / max_area

Y_train = Y_train / max_price
Y_val = Y_val / max_price
Y_test = Y_test / max_price

list_features = ['beds', 'baths', 'area']

train_features = X_train[list_features]
val_features = X_val[list_features]
test_features = X_test[list_features]

BUFFER_SIZE = len(X_train) // 10
BATCH_SIZE = 32

# Descriptive Features Only

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), Y_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), Y_test))
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(input_shape):

    X_input = layers.Input(input_shape)
    X = layers.Dense(32, activation = 'relu')(X_input)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=X_input, outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:])
Path('New York Data/models').mkdir(exist_ok = True)
checkpoint = callbacks.ModelCheckpoint('New York Data/models/feature_model.h5', save_best_only = True, verbose = 1)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 1)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 1)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 1)

model = models.load_model('New York Data/models/feature_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 1, Descriptive Features Only: Test MAPE = {mape:.2f}')

# Descriptive Features + Frontal Images

# Helper Functions 
def load(features, image_file, price):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)[:,:,:3] / 255
    image = tf.cast(image, tf.float32)
    price = tf.cast(price, tf.float32)

    return (features, image), price

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), X_train['image_path'], Y_train))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), X_val['image_path'], Y_val))
val_dataset = val_dataset.map(load)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), X_test['image_path'], Y_test))
test_dataset = test_dataset.map(load)
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(features_input_shape, image_input_shape):

    X_img_input = layers.Input(image_input_shape)
    X = X_img_input
    list_filters = [8,16,32,64,128]
    for n_filters in list_filters:
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Dropout(0.25)(X)
        X = layers.MaxPooling2D((2,2))(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Flatten()(X)

    X = layers.Dense(16, activation = 'relu')(X)

    X_feat_input = layers.Input(features_input_shape)

    X = layers.concatenate([X_feat_input, X])

    X = layers.Dense(32, activation = 'relu')(X)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=[X_feat_input, X_img_input], outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:],(224, 224, 3))
checkpoint = callbacks.ModelCheckpoint('New York Data/models/feature_frontalimg_model.h5', save_best_only = True, verbose = 1)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 1)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 1)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 1)

model = models.load_model('New York Data/models/feature_frontalimg_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 2, Descriptive Features + Frontal Images: Test MAPE = {mape:.2f}')

# Descriptive Features + Frontal Images + Satellite Images

# Helper Functions 
def load(features, image_file, price):
    frontal_image = tf.io.read_file(image_file)
    frontal_image = tf.image.decode_png(frontal_image)[:,:,:3]
    satellite_image = tf.io.read_file(tf.strings.regex_replace(image_file, '/processed_images/','/satellite_images/'))
    satellite_image = tf.image.decode_png(satellite_image)[:,:,:3]
    image = tf.concat([frontal_image, satellite_image],axis=-1) / 255
    image = tf.cast(image, tf.float32)
    price = tf.cast(price, tf.float32)

    return (features, image), price

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), X_train['image_path'], Y_train))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), X_val['image_path'], Y_val))
val_dataset = val_dataset.map(load)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), X_test['image_path'], Y_test))
test_dataset = test_dataset.map(load)
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(features_input_shape, image_input_shape):

    X_img_input = layers.Input(image_input_shape)
    X = X_img_input
    list_filters = [8,16,32,64,128]
    for n_filters in list_filters:
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Dropout(0.25)(X)
        X = layers.MaxPooling2D((2,2))(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Flatten()(X)

    X = layers.Dense(16, activation = 'relu')(X)

    X_feat_input = layers.Input(features_input_shape)

    X = layers.concatenate([X_feat_input, X])

    X = layers.Dense(32, activation = 'relu')(X)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=[X_feat_input, X_img_input], outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:],(224, 224, 6))
checkpoint = callbacks.ModelCheckpoint('New York Data/models/feature_allimg_model.h5', save_best_only = True, verbose = 1)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 1)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 1)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 1)

model = models.load_model('New York Data/models/feature_allimg_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 3, Descriptive Features + Frontal Images + Satellite Images: Test MAPE = {mape:.2f}')

# California Dataset

df = pd.read_pickle('California Data/df.pkl')
df['img_number'] = range(1,len(df)+1)
df['image_path'] = 'California Data/frontal_images/'+df['img_number'].astype(str)+'.jpg'

X_train, X_test, Y_train, Y_test = train_test_split(df,df['price'], test_size = 0.3, random_state = 0)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 0)

max_beds = X_train['beds'].max()
max_baths = X_train['baths'].max()
max_area = X_train['area'].max()
max_price = Y_train.max()

X_train['beds'] = X_train['beds'] / max_beds
X_train['baths'] = X_train['baths'] / max_baths
X_train['area'] = X_train['area'] / max_area

X_val['beds'] = X_val['beds'] / max_beds
X_val['baths'] = X_val['baths'] / max_baths
X_val['area'] = X_val['area'] / max_area

X_test['beds'] = X_test['beds'] / max_beds
X_test['baths'] = X_test['baths'] / max_baths
X_test['area'] = X_test['area'] / max_area

Y_train = Y_train / max_price
Y_val = Y_val / max_price
Y_test = Y_test / max_price

list_features = ['beds', 'baths', 'area']

train_features = X_train[list_features]
val_features = X_val[list_features]
test_features = X_test[list_features]

BUFFER_SIZE = len(X_train) // 10
BATCH_SIZE = 32

# Descriptive Features Only

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), Y_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), Y_test))
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(input_shape):

    X_input = layers.Input(input_shape)
    X = layers.Dense(32, activation = 'relu')(X_input)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=X_input, outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:])
Path('California Data/models').mkdir(exist_ok = True)
checkpoint = callbacks.ModelCheckpoint('California Data/models/feature_model.h5', save_best_only = True, verbose = 0)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 0)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 0)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 0)

model = models.load_model('California Data/models/feature_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 1, Descriptive Features Only: Test MAPE = {mape:.2f}')

# Descriptive Features + Frontal Images

# Helper Functions 
def load(features, image_file, price):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)[:,:,:3] / 255
    image = tf.cast(image, tf.float32)
    price = tf.cast(price, tf.float32)

    return (features, image), price

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), X_train['image_path'], Y_train))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), X_val['image_path'], Y_val))
val_dataset = val_dataset.map(load)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), X_test['image_path'], Y_test))
test_dataset = test_dataset.map(load)
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(features_input_shape, image_input_shape):

    X_img_input = layers.Input(image_input_shape)
    X = X_img_input
    list_filters = [8,16,32,64,128]
    for n_filters in list_filters:
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Dropout(0.25)(X)
        X = layers.MaxPooling2D((2,2))(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Flatten()(X)

    X = layers.Dense(16, activation = 'relu')(X)

    X_feat_input = layers.Input(features_input_shape)

    X = layers.concatenate([X_feat_input, X])

    X = layers.Dense(32, activation = 'relu')(X)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=[X_feat_input, X_img_input], outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:],(224, 224, 3))
checkpoint = callbacks.ModelCheckpoint('California Data/models/feature_frontalimg_model.h5', save_best_only = True, verbose = 1)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 1)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 1)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 1)

model = models.load_model('California Data/models/feature_frontalimg_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 2, Descriptive Features + Frontal Images: Test MAPE = {mape:.2f}')

# Descriptive Features + Frontal Images + Interior Images

# Helper Functions 
def load(features, image_file, price):
    frontal_image = tf.io.read_file(image_file)
    frontal_image = tf.image.decode_png(frontal_image)[:,:,:3]
    bedroom_image = tf.io.read_file(tf.strings.regex_replace(image_file, '/frontal','/bedroom'))
    bedroom_image = tf.image.decode_png(bedroom_image)[:,:,:3]
    bathroom_image = tf.io.read_file(tf.strings.regex_replace(image_file, '/frontal','/bathroom'))
    bathroom_image = tf.image.decode_png(bathroom_image)[:,:,:3]
    kitchen_image = tf.io.read_file(tf.strings.regex_replace(image_file, '/frontal','/kitchen'))
    kitchen_image = tf.image.decode_png(kitchen_image)[:,:,:3]
    image = tf.concat([frontal_image, bedroom_image, bathroom_image, kitchen_image],axis=-1) / 255
    image = tf.cast(image, tf.float32)
    price = tf.cast(price, tf.float32)

    return (features, image), price

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features.to_numpy(), X_train['image_path'], Y_train))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset_length = int(np.ceil(len(X_train)/BATCH_SIZE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_features.to_numpy(), X_val['image_path'], Y_val))
val_dataset = val_dataset.map(load)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset_length = int(np.ceil(len(X_val)/BATCH_SIZE))

test_dataset = tf.data.Dataset.from_tensor_slices((test_features.to_numpy(), X_test['image_path'], Y_test))
test_dataset = test_dataset.map(load)
test_dataset = test_dataset.batch(1)
test_dataset_length = int(np.ceil(len(X_test)/1))

print(f'train_dataset_length: {train_dataset_length}, val_dataset_length: {val_dataset_length}, test_dataset_length: {test_dataset_length}')

def create_model(features_input_shape, image_input_shape):

    X_img_input = layers.Input(image_input_shape)
    X = X_img_input
    list_filters = [8,16,32,64,128]
    for n_filters in list_filters:
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Conv2D(n_filters, 3, activation = 'relu', padding='same')(X)
        X = layers.Dropout(0.25)(X)
        X = layers.MaxPooling2D((2,2))(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Conv2D(list_filters[-1], 3, activation = 'relu', padding='same')(X)
    X = layers.Flatten()(X)

    X = layers.Dense(16, activation = 'relu')(X)

    X_feat_input = layers.Input(features_input_shape)

    X = layers.concatenate([X_feat_input, X])

    X = layers.Dense(32, activation = 'relu')(X)
    X = layers.Dropout(0.25)(X)
    X = layers.Dense(8, activation = 'relu')(X)
    X = layers.Dense(1)(X)

    model = models.Model(inputs=[X_feat_input, X_img_input], outputs = X)
    model.compile('adam',loss='mape')

    return model

epochs = 100
es_patience = epochs // 4
rLR_factor = (1/10)**(0.5)
rLR_patience = epochs // 10

model = create_model(train_features.shape[1:],(224, 224, 12))
checkpoint = callbacks.ModelCheckpoint('California Data/models/feature_allimg_model.h5', save_best_only = True, verbose = 1)
earlystopper = callbacks.EarlyStopping(patience = es_patience, verbose = 1)
reduceLR = callbacks.ReduceLROnPlateau(factor = rLR_factor, patience = rLR_patience, verbose = 1)
hist = model.fit(train_dataset, epochs = 100, validation_data = val_dataset, callbacks = [checkpoint, earlystopper, reduceLR], verbose = 1)

model = models.load_model('California Data/models/feature_allimg_model.h5')
mape = metrics.mean_absolute_percentage_error(Y_test, model.predict(test_dataset, verbose = 0)[:,0]).numpy()
print(f'Stage 3, Descriptive Features + Frontal Images + Interior Images: Test MAPE = {mape:.2f}')