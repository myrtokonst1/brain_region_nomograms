# Constants used to plot the nomograms for both SWA and GPR

# Quantiles shown in the nomograms
percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
quantiles = [percentile/100 for percentile in percentiles]

# Used for gaussian filtering
gaussian_width = 20

# Y-axis limits for different brain regions
hippocampal_volume_y_lim = [2800, 5300]
amygdala_volume_y_lim = [1000, 2400]