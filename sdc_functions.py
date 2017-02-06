import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import cv2
from skimage.feature import hog
import time

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features_img(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
  features = []
  # apply color conversion if other than 'RGB'
  # t = time.time()
  if color_space != 'RGB':
      if color_space == 'HSV':
          feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
      elif color_space == 'LUV':
          feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
      elif color_space == 'HLS':
          feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
      elif color_space == 'YUV':
          feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
      elif color_space == 'YCrCb':
          feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
  else: feature_image = np.copy(image)
  # t2 = time.time()
  # print('Color conv time: %.4f' % (t2 - t))
  # print('image=', feature_image)

  if spatial_feat == True:
      spatial_features = bin_spatial(feature_image, size=spatial_size)
      features.append(spatial_features)
  if hist_feat == True:
      # Apply color_hist()
      hist_features = color_hist(feature_image, nbins=hist_bins)
      features.append(hist_features)
  # t = time.time()
  if hog_feat == True:
  # Call get_hog_features() with vis=False, feature_vec=True
      if hog_channel == 'ALL':
          hog_features = []
          for channel in range(feature_image.shape[2]):
              hog_features.append(get_hog_features(feature_image[:,:,channel],
                                  orient, pix_per_cell, cell_per_block,
                                  vis=False, feature_vec=True))
          hog_features = np.ravel(hog_features)
      else:
          hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                      pix_per_cell, cell_per_block, vis=False, feature_vec=True)
      # Append the new feature vector to the features list
      features.append(hog_features)
      t2 = time.time()
      # print('hog_time %.4f seconds' % (t2-t))
  return np.concatenate(features)


def imread(f):
  image = cv2.imread(f)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        # image = mpimg.imread(file)
        image = imread(file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        file_features = extract_features_img(image, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(file_features)
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    all_extract_time = 0.0
    all_prediction_time = 0.0
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        # features = single_img_features(test_img, color_space=color_space,
        #                     spatial_size=spatial_size, hist_bins=hist_bins,
        #                     orient=orient, pix_per_cell=pix_per_cell,
        #                     cell_per_block=cell_per_block,
        #                     hog_channel=hog_channel, spatial_feat=spatial_feat,
        #                     hist_feat=hist_feat, hog_feat=hog_feat)
        t = time.time()
        features = extract_features_img(test_img, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        t2 = time.time()
        # print('-- extr_feat_time %.4f seconds' % (t2-t))
        all_extract_time += t2 - t

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        t3 = time.time()
        # print('scaler_time %.4f seconds' % (t3-t2))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        t4 = time.time()
        # print('-- prediction_time %.4f seconds' % (t4-t3))
        all_prediction_time += t4 - t3
        # decision = clf.decision_function(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            # decision = clf.decision_function(test_features)
            # print('decision_function = ', decision)
        # elif decision > 0:
        #     print('FILTERED decision_function = ', decision)
        # else:
        #   # DEBUG Purpose
        #   decision = clf.decision_function(test_features)
        #   print('decision_function = ', decision)
    # print('-- extr_feat_time %.4f seconds' % all_extract_time)
    # print('-- prediction_time %.4f seconds' % all_prediction_time)
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list, scale=1):
    # Iterate through list of bboxes
    for ind, box in enumerate(bbox_list):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if isinstance(scale, list):
          heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += scale[ind]
        else:
          heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += scale

    # Return updated heatmap
    return heatmap


def extend_box(box, min_show_box, image):
  newBox = np.copy(box)
  maxX = image.shape[1] - 1
  maxY = image.shape[0] - 1
  spanx = box[1][0] - box[0][0]
  spany = box[1][1] - box[0][1]
  x1 = box[0][0]
  x2 = box[1][0]
  y1 = box[0][1]
  y2 = box[1][1]
  if spanx < min_show_box:
    dx = (min_show_box - spanx)//2
    x1 = x1 - dx
    if x1 < 0: x1 = 0
    x2 = x2 + dx
    if x2 > maxX: x2 = maxX
  if spany < min_show_box:
    dy = (min_show_box - spany)//2
    y1 = y1 - dy
    if y1 < 0: y1 = 0
    y2 = y2 + dy
    if y2 > maxY: y2 = maxY
  return ((x1,y1),(x2,y2))

def find_outer_box(box, hot_windows):
  x1 = box[0][0]
  x2 = box[1][0]
  y1 = box[0][1]
  y2 = box[1][1]
  minX = x1
  maxX = x2
  minY = y1
  maxY = y2
  # print('box =', box)

  for window in hot_windows:
    # print('window =', window)
    xOutside = (window[0][0] > x2 and window[1][0] > x2) or (window[0][0] < x1 and window[1][0] < x1)
    yOutside = (window[0][1] > y2 and window[1][1] > y2) or (window[0][1] < y1 and window[1][1] < y1)
    # print('xout =', xOutside, 'yout =', yOutside)
    if not (xOutside or yOutside):
      # Match
      if window[0][0] < minX:
        minX = window[0][0]
      if window[0][1] < minY:
        minY = window[0][1]
      if window[1][0] > maxX:
        maxX = window[1][0]
      if window[1][1] > maxY:
        maxY = window[1][1]
      # print('match: ', ((minX, minY), (maxX, maxY)))
  # print('res: ', ((minX, minY), (maxX, maxY)))
  return ((minX, minY), (maxX, maxY))

def get_outer_bboxes(labels, hot_windows):
  bboxes = []
  min_box = 20
  show_box = 96
  for car in range(1, labels[1] + 1):
    nonzero = (labels[0] == car).nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    bbox = find_outer_box(bbox, hot_windows)
    # bbox = extend_box(bbox, show_box, labels[0])
    bboxes.append(bbox)
  return bboxes


def draw_labeled_bboxes(img, labels, hot_windows):
    draw_image = np.copy(img)
    bboxes = get_outer_bboxes(labels, hot_windows)
    draw_image = draw_boxes(img, bboxes, color = (0,255,0), thick = 6)
    # for box in bboxes:
    #   cv2.rectangle(draw_image, box[0], box[1], (0,255,0), 6)
    # Return the image
    return draw_image, bboxes

def intersection(box1, box2):
    x = max(box1[0][0], box2[0][0])
    y = max(box1[0][1], box2[0][1])
    w = min(box1[1][0], box2[1][0]) - x
    h = min(box1[1][1], box2[1][1]) - y
    if w < 0 or h < 0: return 0
    return w*h

def combine_boxes(prev_box, new_box, prev_factor = 0.9):
    # print('prev_box =', prev_box)
    # print('new_box =', new_box)
    minX = int(round(prev_box[0][0] * prev_factor + new_box[0][0] * (1. - prev_factor)))
    minY = int(round(prev_box[0][1] * prev_factor + new_box[0][1] * (1. - prev_factor)))
    maxX = int(round(prev_box[1][0] * prev_factor + new_box[1][0] * (1. - prev_factor)))
    maxY = int(round(prev_box[1][1] * prev_factor + new_box[1][1] * (1. - prev_factor)))
    # print('comb_res =', ((minX, minY),(maxX, maxY)))
    return ((minX, minY),(maxX, maxY))


def find_max_intersection(cars_list, box):
    max_v = 0
    max_ind = -1
    # For car
    for ind, car_box in enumerate(cars_list):
        inter = intersection(car_box, box)
        if inter > max_v:
            max_v = inter
            max_ind = ind
    return max_v, max_ind


def combine_with_prev(prev, prev_age, curr, prev_factor=0.8, fresh_age = 2):
    new_cars = []
    new_cars_age = []
    prev_cars = prev[:]
    prev_cars_age = prev_age[:]
    curr_cars = curr[:]
    fresh_car_age = fresh_age

    while len(prev_cars) > 0:
        prev_car = prev_cars.pop()
        prev_car_age = prev_cars_age.pop()
        inter_v, inter_ind = find_max_intersection(curr_cars, prev_car)
        if inter_v > 0:
            # Found car in a curr list
            new_cars.append(combine_boxes(prev_car, curr_cars[inter_ind], prev_factor=prev_factor))
            new_cars_age.append(fresh_car_age)
            del curr_cars[inter_ind]
        else:
            # Not found in curr list, check for age
            if prev_car_age > 0:
                # Add and decrease age (ttl:)
                new_cars.append(prev_car)
                new_cars_age.append(prev_car_age - 1)
            else:
                print('dropped car ', prev_car)


    # Add all other current cars to the list of new cars
    while len(curr_cars) > 0:
        curr_car = curr_cars.pop()
        new_cars.append(curr_car)
        new_cars_age.append(fresh_car_age)

    return new_cars, new_cars_age

def cars_search_windows(image, prev_cars):
  maxX, maxY = image.shape[1] - 1, image.shape[0] - 1
  cars_windows = []
  window_sizes = [256, 224, 192, 160, 128, 96, 64] # [256, 224, 192, 160, 128, 96, 64]
  for cw in prev_cars:
    # print('cw =', cw)
    cX = (cw[1][0] + cw[0][0])//2
    cY = (cw[1][1] + cw[0][1])//2
    # print('cX =', cX, ' cY=', cY)
    for wind in window_sizes:
      x1 = cX - wind//2
      x2 = cX + wind//2
      y1 = cY - wind//2
      y2 = cY + wind//2
      # Clip
      if x1 < 0: x1 = 0
      if x2 > maxX: x2 = maxX
      if y1 < 0: y1 = 0
      if y2 > maxY: y2 = maxY
      # print('app ', ((x1,y1),(x2,y2)))
      cars_windows.append(((x1,y1),(x2,y2)))
  return cars_windows


'''
def combine_with_prev(prev, prev_age, curr, prev_factor=0.8):
    new_cars = []
    prev_cars_remained = prev[:]
    prev_cars_age_remained = prev_age[:]
    new_cars_age = []
    # For each new car
    for curr_ind, curr_car_box in enumerate(curr):
        max_v = 0
        max_ind = -1
        # Iterate over previous cars
        for prev_ind, prev_car_box in enumerate(prev):
            inter = intersection(curr_car_box, prev_car_box)
            if inter > max_v:
                max_v = inter
                max_ind = prev_ind
        # Combine with the car that has maximum intersection
        if max_v > 0:
            # print(curr_car_box, 'match', max_ind)
            new_cars.append(combine_boxes(prev[max_ind], curr_car_box, prev_factor=prev_factor))
            # Remove from remained list
            del prev_cars_remained[max_ind]
            del prev_cars_age_remained[max_ind]
        else:
            new_cars.append(curr_car_box)
        # Add age 0 as a fresh cars
        new_cars_age.append(2)
    # Add prev cars remained if their age > 0
    for prev_remained, prev_remained_age in zip(prev_cars_remained, prev_cars_age_remained):
      # If car is not older than 2 frame - add and make it older
      if prev_remained_age > 0:
        new_cars.append(prev_remained)
        new_cars_age.append(prev_remained_age)

    return new_cars, new_cars_age
'''

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def save_output_img(img, fname):
  # Make it work for bin images too
  if len(img.shape) == 2:
      imgN = bin_to_rgb(img)
  else:
      imgN = img
  misc.imsave('output_images/output_%s.png' % fname, imgN)

def get_fig_image(fig):
  # http://www.itgo.me/a/1944619462852588132/matplotlib-save-plot-to-numpy-array
  # This is very BAD hack ....
  fig.savefig('output_images/tmp.png')
  # fig.canvas.draw()
  # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  # plt.close(fig)
  plt.close(fig)
  img = cv2.imread('output_images/tmp.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # from scipy import misc
  # img = misc.imread('output_images/temp.png')
  return img

def bin_to_rgb(bin_image):
  return cv2.cvtColor(bin_image*255, cv2.COLOR_GRAY2RGB)

def compose_images(dst, src, nrows, ncols, num):
  assert 0 < num <= nrows * ncols

  if nrows > ncols:
      newH = int(dst.shape[0]/nrows)
      dim = (int(dst.shape[1] * newH/dst.shape[0]), newH)
  else:
      newW = int(dst.shape[1]/ncols)
      dim = (newW, int(dst.shape[0] * newW/dst.shape[1]))

  # Make it work for bin images too
  if len(src.shape) == 2:
      srcN = bin_to_rgb(src)
  else:
      srcN = np.copy(src)

  img = cv2.resize(srcN, dim, interpolation = cv2.INTER_AREA)
  nr = (num - 1) // ncols
  nc = (num - 1) % ncols
  dst[nr * img.shape[0]:(nr + 1) * img.shape[0], nc * img.shape[1]:(nc + 1) * img.shape[1]] = img
  return dst


# Normalize data
def normalize(data):
    pixel_depth = 255
    return (data - pixel_depth / 2) / pixel_depth
