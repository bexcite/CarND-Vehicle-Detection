'''
  Config common for train and inference
'''

class Config(object):
  color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 18  # HOG orientations
  pix_per_cell = 6 # HOG pixels per cell
  cell_per_block = 2 # HOG cells per block
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
  spatial_size = (24, 24) # Spatial binning dimensions
  hist_bins = 16    # Number of histogram bins
  spatial_feat = True # Spatial features on or off
  hist_feat = True # Histogram features on or off
  hog_feat = True # HOG features on or off
  y_start_stop = [300, 720] # Min and max in y to search in slide_window()