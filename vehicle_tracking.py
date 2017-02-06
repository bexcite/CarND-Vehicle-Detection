'''
  Track Vehicle on the video
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import time
import os
import pickle
from moviepy.editor import VideoFileClip
# from sklearn.svm import LinearSVC
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog
from sdc_functions import *
from scipy.ndimage.measurements import label
# from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config

class VehicleTracker(object):
  def __init__(self, svc, scaler, config):
    self.svc = svc
    self.scaler = scaler
    self.config = config
    self.all_windows = []
    self.prev_car_bboxes = []
    self.prev_car_ages = []
    self.counter = 0
    self.save_time = time.strftime("%Y%m%d%H%M%S")

  def process_image(self, image):
    config = self.config
    window_sizes = [160, 128, 96] # [256, 224, 192, 160, 128, 96, 64]
    window_sizes_box = [(b, b) for b in window_sizes]
    overlap_factor_x = self.config.overlap_factor_x
    overlap_factor_y = self.config.overlap_factor_y
    svc = self.svc
    scaler = self.scaler

    if len(self.all_windows) == 0:
      # Calculate all_windows positions (only once)
      t0 = time.time()
      for widx, window_size in enumerate(window_sizes_box):
          windows = slide_window(image.shape, x_start_stop=[None, None], y_start_stop=config.y_start_stop,
                          xy_window=window_size, xy_overlap=(overlap_factor_x, overlap_factor_y))
          self.all_windows.extend(windows)
      print('all_windows len = %d, Time: %.4f seconds' % (len(self.all_windows), (time.time() - t0)))



    # Prepare empty resulting image
    resImg = np.zeros_like(image)

    # Show: Original Image
    compose_images(resImg, image, 2, 2, 3)

    # Slide windows over image and classify them
    all_hot_windows = search_windows(image, self.all_windows, svc, scaler, color_space=config.color_space,
                            spatial_size=config.spatial_size, hist_bins=config.hist_bins,
                            orient=config.orient, pix_per_cell=config.pix_per_cell,
                            cell_per_block=config.cell_per_block,
                            hog_channel=config.hog_channel, spatial_feat=config.spatial_feat,
                            hist_feat=config.hist_feat, hog_feat=config.hog_feat)

    # Additionally add search windows from previous car position to better search in previous locations
    prev_cars_windows = cars_search_windows(image, self.prev_car_bboxes)

    ## Draw Prev cars windows Boxes
    draw_image = np.copy(image)
    pwindows_img = draw_boxes(draw_image, prev_cars_windows, color=(255, 0, 0), thick=3)
    compose_images(resImg, pwindows_img, 2, 2, 3)


    # print('prev_cars_wind = ', len(prev_cars_windows))
    prev_cars_hot_windows = search_windows(image, prev_cars_windows, svc, scaler, color_space=config.color_space,
                            spatial_size=config.spatial_size, hist_bins=config.hist_bins,
                            orient=config.orient, pix_per_cell=config.pix_per_cell,
                            cell_per_block=config.cell_per_block,
                            hog_channel=config.hog_channel, spatial_feat=config.spatial_feat,
                            hist_feat=config.hist_feat, hog_feat=config.hog_feat)
    # print('prev_cars_hot = ', len(prev_cars_hot_windows))
    all_hot_windows.extend(prev_cars_hot_windows)


    ## Draw All Boxes
    draw_image = np.copy(image)
    windows_img = draw_boxes(draw_image, all_hot_windows, color=(0, 0, 255), thick=2)
    compose_images(resImg, windows_img, 2, 2, 1)

    ## Heat Map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, all_hot_windows)

    ## Draw Heat Map 1
    final_map = np.clip(heat, 0, 255)
    fig = plt.figure(figsize=(10, 5))
    plt.title('Start')
    plt.imshow(final_map, cmap='hot')
    data = get_fig_image(fig)
    compose_images(resImg, data, 4, 4, 3)


    # Add previous frame cars and factor their age
    heat = add_heat(heat, self.prev_car_bboxes, scale = self.prev_car_ages)

    ## Draw Heat Map 2
    final_map = np.clip(heat, 0, 255)
    fig = plt.figure(figsize=(10, 5))
    plt.title('With Prev Car')
    plt.imshow(final_map, cmap='hot')
    data = get_fig_image(fig)
    compose_images(resImg, data, 4, 4, 4)


    heat = apply_threshold(heat, 3)

    ## Draw Heat Map 3
    final_map = np.clip(heat, 0, 255)
    fig = plt.figure(figsize=(10, 5))
    plt.title('Thresholded')
    plt.imshow(final_map, cmap='hot')
    data = get_fig_image(fig)
    compose_images(resImg, data, 4, 4, 7)


    labels = label(heat)

    ## Draw Label Map
    fig = plt.figure(figsize=(10, 5))
    plt.title('%d cars found' % labels[1])
    plt.imshow(labels[0], cmap='gray')
    data = get_fig_image(fig)
    compose_images(resImg, data, 4, 4, 8)


#     print(labels[1], 'cars found')
#     plt.subplot(1, 2, 2)

    ## Dtaw Labels Map
#     axs[idx, 2].set_title('%d cars found' % labels[1])
#     axs[idx, 2].imshow(labels[0], cmap='gray')

    # Get Car Bboxes
    car_bboxes = get_outer_bboxes(labels, all_hot_windows)

    # Combine Prev and Current
    self.prev_car_bboxes, self.prev_car_ages = combine_with_prev(self.prev_car_bboxes,
        self.prev_car_ages, car_bboxes, prev_factor=0.8, fresh_age = 2)

    ## Draw Outer Box
    bbox_image = draw_boxes(image, self.prev_car_bboxes, color = (0,255,0), thick = 6)
    # bbox_image, car_bboxes = draw_labeled_bboxes(image, labels, all_hot_windows)
    compose_images(resImg, bbox_image, 2, 2, 4)

    # Save test image
    if self.counter % 5 == 0:
      save_output_img(resImg, "%s_%4d" % (self.save_time, self.counter))

    self.counter += 1

    return resImg



def main():
  parser = argparse.ArgumentParser(description="Vehicle Tracking on a video")
  parser.add_argument('--video', type=str, default='project_video.mp4', help='project video')
  parser.add_argument('--output_video', default='output.mp4', type=str, help='output video')
  parser.add_argument('--svc_scaler', type=str, default='pickle_svc_scaler.p', help='saved file with SVC and Scaler')
  parser.add_argument('--verbose', default=False, action='store_true', help='verbosity flag')
  parser.add_argument('--t_start', type=float, default=0.0, help='t_start param')
  parser.add_argument('--t_end', type=float, default=0.0, help='t_end param')

  args = parser.parse_args()

  video_file = args.video
  output_video_file = args.output_video
  svc_scaler = args.svc_scaler
  verbose = args.verbose
  t_start = args.t_start
  t_end = args.t_end

  print("Video file: {}".format(video_file))
  print("Output video file: {}".format(output_video_file))
  print("SVC_scaler file: {}".format(svc_scaler))
  print("t_start: {}".format(t_start))
  print("t_end: {}".format(t_end))
  print("Verbose: {}".format(verbose))

  print("Vehicle Tracking ...")

  ## Restore
  restore_data = pickle.load(open('pickle_svc_scaler.p', "rb" ))
  svc = restore_data["svc"]
  scaler = restore_data["scaler"]

  # Get common config
  config = Config()

  clip = VideoFileClip(video_file)
  if t_end > 0.0:
    clip = clip.subclip(t_start=t_start, t_end=t_end)
  else:
    clip = clip.subclip(t_start=t_start)

  # sampling = 6./(clip.duration * 25) if verbose else 0
  vehicleTracker = VehicleTracker(svc=svc, scaler=scaler, config=config)
  clip = clip.fl_image(vehicleTracker.process_image)
  clip.write_videofile(output_video_file, audio=False)

if __name__ == '__main__':
  main()
