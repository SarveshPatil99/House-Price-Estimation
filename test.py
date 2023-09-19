from tensorflow.keras import models
import numpy as np
import cv2

print('New York example:')
with open('test_data/NY/features.txt', 'r') as f:
    features_text = f.read()
with open('test_data/NY/train_stats.txt', 'r') as f:
    train_stats_text = f.read()
features = [int(f) for f in features_text.split(' ')[:-1]]
train_stats = [float(f) for f in train_stats_text.split(', ')]
target = int(features_text.split(' ')[-1])
frontal_img = cv2.imread('test_data/NY/30691644.png')[:, :, ::-1]/255
sat_img = cv2.imread('test_data/NY/30691644_sat.png')[:, :, ::-1]/255
print(f'Actual Price: {target:.2f}')
model = models.load_model('final_models/NY/feature_model.h5')
pred = (model.predict((np.array(features) / np.array(train_stats[:-1])).reshape(1, -1),
                      verbose=0) * train_stats[-1])[0, 0]
print(f'Feature Model Prediction: {pred:.2f}')
model = models.load_model('final_models/NY/feature_frontalimg_model.h5')
pred = (model.predict([(np.array(features) / np.array(train_stats[:-1])).reshape(1, -1),
                       frontal_img.reshape(1, 224, 224, 3)], verbose=0) * train_stats[-1])[0, 0]
print(f'Feature+Frontal Model Prediction: {pred:.2f}')
model = models.load_model('final_models/NY/feature_allimg_model.h5')
pred = (model.predict([(np.array(features) / np.array(train_stats[:-1])).reshape(1, -1),
                       np.concatenate([frontal_img, sat_img], axis=-1).reshape(1, 224, 224, 6)],
                      verbose=0) * train_stats[-1])[0, 0]
print(f'Feature+Frontal+Satellite Model Prediction: {pred:.2f}')

print('\nCalifornia example:')
with open('test_data/Cali/features.txt', 'r') as f:
    features_text = f.read()
with open('test_data/Cali/train_stats.txt', 'r') as f:
    train_stats_text = f.read()
features = [int(f) for f in features_text.split(' ')[:-1]]
train_stats = [float(f) for f in train_stats_text.split(', ')]
target = int(features_text.split(' ')[-1])
frontal_img = cv2.imread('test_data/Cali/156_frontal.jpg')[:, :, ::-1]/255
bed_img = cv2.imread('test_data/Cali/156_bed.jpg')[:, :, ::-1]/255
bath_img = cv2.imread('test_data/Cali/156_bath.jpg')[:, :, ::-1]/255
kitchen_img = cv2.imread('test_data/Cali/156_kitchen.jpg')[:, :, ::-1]/255
print(f'Actual Price: {target:.2f}')
model = models.load_model('final_models/Cali/feature_model.h5')
pred = (model.predict((np.array(features)/np.array(train_stats[:-1])).reshape(1, -1), verbose=0)*train_stats[-1])[0, 0]
print(f'Feature Model Prediction: {pred:.2f}')
model = models.load_model('final_models/Cali/feature_frontalimg_model.h5')
pred = (model.predict([(np.array(features)/np.array(train_stats[:-1])).reshape(1, -1),
                       frontal_img.reshape(1, 224, 224, 3)], verbose=0)*train_stats[-1])[0, 0]
print(f'Feature+Frontal Model Prediction: {pred:.2f}')
model = models.load_model('final_models/Cali/feature_allimg_model.h5')
pred = (model.predict([(np.array(features)/np.array(train_stats[:-1])).reshape(1, -1),
                       np.concatenate([frontal_img, bed_img, bath_img, kitchen_img], axis=-1)
                      .reshape(1, 224, 224, 12)], verbose=0)*train_stats[-1])[0, 0]
print(f'Feature+Frontal+Interior Model Prediction: {pred:.2f}')
