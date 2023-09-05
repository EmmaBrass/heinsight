import os
import cv2
import time
from heinsight.liquidlevel.liquid_level import LiquidLevel
from heinsight.liquidlevel.track_tolerance_levels import \
    TrackLiquidToleranceLevels, TrackOneLiquidToleranceLevel, \
    TrackTwoLiquidToleranceLevels

# OLD CODE - HEINSIGHT EDGE DETECTION METHODS

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
    use_reference=False,
    rows_to_count=10,
    number_of_liquid_levels_to_find=1,
    find_meniscus_minimum=0.1,
    no_error=False,
    liquid_level_data_save_folder=os.path.join(os.path.abspath(os.path.curdir),
        'logs')
    )  

start_image = cv2.imread("C:\\Users\\eb559981\\Documents\\Images\\IMG_5193.jpeg")
distill_volumes = 4
threshold = 0.01

liquid_level.start(image=start_image, select_region_of_interest=True, 
    set_reference=False, volumes_list = ['4'], select_tolerance=False)

run_image = cv2.imread("C:\\Users\\eb559981\\Documents\\Images\\IMG_5193.jpeg")

liquid_level.run(input_image=run_image) 