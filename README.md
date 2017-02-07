# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal is to build a vehicle tracking pipeline using Histogram of Oriented Gradients (HOG) features and trained Linear SVM classifier.

## Labeled Data

Data used in the project:
 1. [Vehicle image set](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
 2. [Non-vehicle image set](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

 These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.  

## Histogram of Oriented Gradients

Params of HOG feature extractor was selected together with SVM Classifier so the extracted features lead to the better accuracy ov SVC. Details about Training SVC see below.

The best results on SVM classifier was received on the next parameters:
```
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```
Increased number of orientations (`orient = 18`) helps capture more variations in shapes. Combining ALL color channels increased the final accuracy as well. Color spaces HLS, LUV and HSV was all good but on HLS I've got couple of points better accuracy on test data (**99.32%**).

HOG feature extraction code could be found in `sdc_functions.py` method
```
get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
```

## Train SVM Classifier

Additionally to HOG features I've also added color histogram (with 16 bins) and spatial features by scaling an image to (24,24) pixels. So the resulting feature vector was HOG 3 channels + color histogram + spatial with the total number of params **12,360**. All feature was scaled by `sklearn.preprocessing.StandardScaler` so to have zero mean and unit variance.

I've also split the all training data to train and test sets (20%). The resulting training set contained 14,208 labeled images and test set – 3,552 images.

Training time ~ 20 seconds and feature extraction for all dataset ~ 116 seconds on MacBook Pro 13'.

Test accuracy of trained Linear SVC is **99.3243%**

SVC classifier and scaler saved to `pickle_svc_scaler.p` file and used later in `vehicle_tracking.py`

Below is an example of images from test set with prediction and a value of decision function:

<<IMAGE>>

To train classifier on your machine first modify `train_scv.py` params (lines 17-22):
```
output_file = 'pickle_svc_scaler1.p' # Changed so to avoid accidental file overwrite
test_size = 0.2
sample_size = 500 # 0 - ALL

vehicles_folder = '../vehicles'
non_vehicles_folder = '../non-vehicles'
```

Then run following command:
```
python train_scv.py
```

## Sliding window search

In order to find cars on the image I've split the whole image on the series of overlapping windows of three different sizes `[(96,96), (128, 128), (144, 144)]` that scan image from `ymin = 400` to `ymax = 620`. Such limitation decreases the total amount of windows which is quicker to compute later and decreases the false positive ration by removing trees and road signs.

I've discovered that distant cars are not recognizing well so I've later added 2 more window sizes (`(72,72)` and `(48, 48)`) for two more distant regions.

So the final configuration of sliding windows looks like this (from `config.py`):
```
window_sizes = [144, 128, 96]

far_window1 = (
  (200, y_start_stop[0]),
  (1180, (y_start_stop[0] + y_start_stop[1])//2)
)
far_window2 = (
  (300, y_start_stop[0]),
  (1080, y_start_stop[0] + int((y_start_stop[1] - y_start_stop[0])*0.4))
)
far_window1_size = 72
far_window2_size = 48
```

Below is the image with all windows and specific distant regions:

<<IMAGE>>


## Thresholding and Vehicle Tracking

In order to recognize a car on a heatmap I've applied threshold <= 2, so it should be at least 3 window that _votes_ for a car.

### Tracking vehicle from previous frame

Successfully recognized vehicles on one frame additionally adds a couple of points to a heatmap on the next frame based on vehicle _age_ (it's actually TTL but in code I've called it as age). Each new or confirmed car (thresholded and labeled region) receive age (TTL) 2 that decreases by one each frame unless it's not confirmed on the next frame.

It means that for a once detected car we need to have at least one detected window on a next frame that together with a previous car box forms 3 points on a heatmap that will lead to confirmed car.

If car is not confirmed during next 2 frames it will be dropped on a third (age < 0). Also aging cars contribute less to the confirmation so if car missed one confirmation it's age becomes 1 and on the next frame it adds only 1 point to a heat map. Consequently if car reaches age 0 it's almost _dead_ unless we will have 3 more fresh windows that can vote for that car.

### Look for previous car locations more carefully

Another mechanism that helps car to stay alive is additional search windows that are generated dynamically around previous cars and added to an all windows for a pattern search and classification. It helps to add couple of activated images and prevents car from disappearing if standard grid failed to set the right position for the car to be detected.

Each car generates additionally 9 * 5 = 40 windows.

```
# Generates additional windows for previous cars
def cars_search_windows(image, prev_cars):
  maxX, maxY = image.shape[1] - 1, image.shape[0] - 1
  cars_windows = []
  window_sizes = [128, 96, 80, 64, 48] # [256, 224, 192, 160, 128, 96, 64]
  for cw in prev_cars:
    wins = get_detailed_windows(image, cw, window_sizes)
    cars_windows.extend(wins)
  return cars_windows

# Generates additional windows around one car - 9 point
def get_detailed_windows(image, car_box, window_sizes):

    cX = (car_box[1][0] + car_box[0][0])//2
    cY = (car_box[1][1] + car_box[0][1])//2

    # Get the half of the smallest window
    delta = int(window_sizes[-1] * 0.5)

    # Shift window to delta in each direction
    deltak = [
        (-1,-1), (0,-1), (1,-1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ]

    # Iterate over all window_sizes and all positions
    windows = []
    for wind in window_sizes:
        for k in deltak:
            kx, ky = k[0], k[1]
            cx = int(cX + kx * delta)
            cy = int(cY + ky * delta)
            w = get_window_around(image, (cx, cy), wind)
            windows.append(w)
    return windows

```

Here how such additional windows looks on a visualization:

<<IMAGE of the rose>>

### Merge and combine previous and new cars

In order to continue tracking the same car I'm combining new car windows with previous based on the intersection area. So for each previous car look for the most intersected new box and combine them with a factor `0.9` (`prev_car_box * 0.9 + new_car_box * 0.1`). Such combination smoothening the car box movement between frames and represents the gradual change in a car location rather than big jumps of the box around.

Sometimes `label` functions splits the heatmap into two cars which leads into appearing of the ghost cars. To avoid this I've used the merge function that looks for an intersection areas between the resulting car_boxes and merges boxes that intersects more than 70%.

Below is the function that merges car in a list:
```
# Look for interlaping car boxes and merge them
def merge_combine_cars_list(cars_list, cars_age_list, prev_factor = 0.8, merge_overlap = 0.8):

  # Copy lists
  new_cars = cars_list[:]
  new_cars_age = cars_age_list[:]

  new_cars_merged = []
  new_cars_merged_age = []

  while len(new_cars) > 0:
    car = new_cars.pop()
    car_age = new_cars_age.pop()
    inter_v, inter_ind = find_max_intersection(new_cars, car)
    while inter_v > 0:
      if inter_v > merge_overlap * box_area(new_cars[inter_ind]):
        # Car main
        car = combine_boxes(car, new_cars[inter_ind], prev_factor=prev_factor)
        car_age = max(car_age, new_cars_age[inter_ind])
        del new_cars[inter_ind]
        del new_cars_age[inter_ind]
        inter_v, inter_ind = find_max_intersection(new_cars, car)
      elif inter_v > merge_overlap * box_area(car):
        # New_cars main
        car = combine_boxes(new_cars[inter_ind], car, prev_factor=prev_factor)
        car_age = max(car_age, new_cars_age[inter_ind])
        del new_cars[inter_ind]
        del new_cars_age[inter_ind]
        inter_v, inter_ind = find_max_intersection(new_cars, car)
      else:
        # No more valuable intersections for the current car
        break

    new_cars_merged.append(car)
    new_cars_merged_age.append(car_age)

  return new_cars_merged, new_cars_merged_age

```

## Video result

The final result that was received on `project_video.mp4`

<< VIDEO>>

## Run Vehicle Tracking

In order to run `vehicle_tracking.py` on your machine use the next command:
```
python vehicle_tracking.py --output output.mp4 --t_start 0.0 --t_end 50.0
```
be sure that you have the latest and not modified `pickle_svc_scaler.p` or at least trained on a full dataset. Add `--verbose` flag to turn on sample outputs from the pipeline into `output_images` folder.

## Discussion

There a lot of room for improvements:

1. The biggest drawback of the implementation is speed. Time to process one frame is ~7-9 sec on MacBook Pro 13' and this is very slow. Should invest more time later for optimization. Ideas: optimize window sizes and overlapping factor, more carefully select region and make it adjust dynamically so further parts of the road scans with smaller size filter and closer parts scan with bigger size.

2. Combine line detection with dynamic area selection for each window size. It will help to reduce total number of windows to scan.

3. There still some false positive on the shades under the trees. To solve this we can add more training samples to our classifier so it can recognize such situations.

4. Possible use of CNN as a search and classification algorithm for cars.

5. Kalman filters should work too for tracking vehicle between frames, but I don't have experience with them yet so it's more research task :)

6. Add params to `train_scv.py` so we can run it from command line without changing code.

7. Add more code separation and structure.
