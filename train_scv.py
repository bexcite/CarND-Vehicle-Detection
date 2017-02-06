'''
  Train Classifier to detect cars
'''
import numpy as np
import cv2
import glob
import time
import os
import pickle
from config import Config
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sdc_functions import *
from sklearn.model_selection import train_test_split

output_file = 'pickle_svc_scaler1.p'
test_size = 0.2
sample_size = 500 # 0 - ALL

vehicles_folder = '../vehicles'
non_vehicles_folder = '../non-vehicles'

vehicles = glob.glob(vehicles_folder + '/**/*.png', recursive=True)
print('vehicles = ', len(vehicles))

non_vehicles = glob.glob(non_vehicles_folder + '/**/*.png', recursive=True)
print('non_vehicles = ', len(non_vehicles))

### Subsample

if sample_size > 0:
  vehicles = np.random.choice(vehicles, sample_size)
  non_vehicles = np.random.choice(non_vehicles, sample_size)
  print('Sampled to %d images.' % sample_size)

# Initialize config with all parameters
config = Config()


print('Extracting features ...')
t=time.time()
car_features = extract_features(vehicles, color_space=config.color_space,
                        spatial_size=config.spatial_size, hist_bins=config.hist_bins,
                        orient=config.orient, pix_per_cell=config.pix_per_cell,
                        cell_per_block=config.cell_per_block,
                        hog_channel=config.hog_channel, spatial_feat=config.spatial_feat,
                        hist_feat=config.hist_feat, hog_feat=config.hog_feat)
notcar_features = extract_features(non_vehicles, color_space=config.color_space,
                        spatial_size=config.spatial_size, hist_bins=config.hist_bins,
                        orient=config.orient, pix_per_cell=config.pix_per_cell,
                        cell_per_block=config.cell_per_block,
                        hog_channel=config.hog_channel, spatial_feat=config.spatial_feat,
                        hist_feat=config.hist_feat, hog_feat=config.hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
t2=time.time()

print('Feature extracted :')
print('scaled_X =', len(scaled_X))
print('y =', len(y))
print('feature length = ', len(scaled_X[0]))
print('Extraction time %.2f seconds' % (t2-t))



### Train SVC

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=test_size, random_state=rand_state)

print('X_train =', len(X_train));
print('X_test =', len(X_test));
print('y_train =', len(y_train));
print('y_test =', len(y_test));

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
print('Train SVC classifier ...')
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test)*100, 4))

## Store SVC classifier and scaler to file
store_data = {"svc":svc, "scaler": X_scaler}
with open(output_file, 'wb') as f:
    pickle.dump(store_data, f)
print("SVC and Scaler stored to '%s' file" % output_file)
