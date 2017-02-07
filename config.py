'''
  Config common for train and inference
'''

class Config(object):
  color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 18  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  cell_per_block = 2 # HOG cells per block
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
  spatial_size = (24, 24) # Spatial binning dimensions
  hist_bins = 16    # Number of histogram bins
  spatial_feat = True # Spatial features on or off
  hist_feat = True # Histogram features on or off
  hog_feat = True # HOG features on or off
  y_start_stop = [400, 620] # Min and max in y to search in slide_window()
  overlap_factor_x = 0.75
  overlap_factor_y = 0.75

  window_sizes = [144, 128, 96] # [256, 224, 192, 160, 128, 96, 64]

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


'''
Good params
  color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 18  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  cell_per_block = 2 # HOG cells per block
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
  spatial_size = (24, 24) # Spatial binning dimensions
  hist_bins = 16    # Number of histogram bins
  spatial_feat = True # Spatial features on or off
  hist_feat = True # Histogram features on or off
  hog_feat = True # HOG features on or off
  y_start_stop = [300, 720] # Min and max in y to search in slide_window()

'''
