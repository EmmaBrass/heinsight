import os
import cv2
import time
from heinsight.liquidlevel.liquid_level import LiquidLevel
from heinsight.liquidlevel.track_tolerance_levels import \
    TrackLiquidToleranceLevels, TrackOneLiquidToleranceLevel, \
    TrackTwoLiquidToleranceLevels

# get image
# select ROI as normal
# reduce ROI to a vertical line of pixels, with the values as the 
# average for that row in the ROI
# make a scatter plot of I vs h
# find second derivative at each point on the graph
# select the two points with the greatest absolute second derivative
# ensure that these two points are close enough together - within a threshold
# e.g. could do select the greatest one and then select the next greatest 
# within the threshold area
# set liquid level line as being halfway between these two points

# try this for greyscale at first and then try other channels like RBG ?

def resize(input_image):
    h, w = input_image.shape[:2]
    aspect = h/w
    new_width = 600
    new_height = int(new_width*aspect)
    output_image = cv2.resize(input_image, dsize=(new_width,new_height))
    return output_image

tracker = TrackTwoLiquidToleranceLevels()
liquid_level = LiquidLevel(
    camera=None,
    track_liquid_tolerance_levels=tracker,
    use_tolerance=False,
    use_reference=True,
    rows_to_count=10,
    number_of_liquid_levels_to_find=1,
    find_meniscus_minimum=0.1,
    no_error=False,
    liquid_level_data_save_folder=os.path.join(os.path.abspath(os.path.curdir),
        'logs')
    )  

#start_image = cv2.imread("/Users/Emma/Documents/Documents - MacBook Pro/Liverpool PhD/Code/heinsight/images/Fume Hood 7-24-2023, 5-15-13 PM.jpg")
#distill_volumes = 4
#threshold = 0.01

# liquid_level.start(image=start_image, select_region_of_interest=True, 
#     set_reference=True, volumes_list = ['4'], select_tolerance=False)

run_image = cv2.imread("/Users/Emma/Documents/Documents - MacBook Pro/Liverpool PhD/Code/heinsight/images/Fume Hood 7-24-2023, 5-15-13 PM.jpg")

liquid_level.test_run(input_image=run_image) 

# if percent_diff < threshold:
#     print("above threshold, would keep heating")
# else:
#     print("below theshold, would move on to next step")
