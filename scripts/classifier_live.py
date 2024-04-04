# 
#  Copyright (C) 2023 Istituto Italiano di Tecnologia (IIT)
#  
#  This software may be modified and distributed under the terms of the
#  GPL-2+ license. See the accompanying LICENSE file for details.
# 

import argparse
import cv2
from os.path import join
from digit_interface.digit import Digit
from surfaceclassifier.digit_utilities.prune_point_cloud_class import PrunePointCloud


def function_loop(prune_object, digit_object):
    """
    Function to be passed in the loop to update the point cloud dependending on the image coming from the DIGIT sensor.

    Parameters
    ----------
    prune_object : PrunePointCloud
        wrapper for the point cloud pruner
    digit_object : Digit
        wrapper for the Digit sensor
    """
    
    rgb = cv2.cvtColor(digit_object.get_frame(), cv2.COLOR_BGR2RGB)

    return prune_object.get_point_cloud(rgb)

def main():
    """
    Main function of the script.
    """

    # Initialize the parser and parse the inputs
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--weights', dest='weights', help='absolute path of the weights of the model', type=str, required=True)
    parser.add_argument('--digit_number', dest='digit_number', help='serial number of the digit', type=str, required=True)
    parser.add_argument('--repo_path', dest='repo_path', help='absolute path to the repo', type=str, required=True)
    args = parser.parse_args()
    point_cloud_prune = PrunePointCloud(join(args.repo_path, 'surfaceclassifier/src_compare_dataset/config/dino_v2_config.yaml'),
                                        args.weights,
                                        join(args.repo_path, 'config/config.ini'))

    digit_handler = Digit(args.digit_number)
    digit_handler.connect()

    point_cloud_prune.get_class_live(digit_handler)




main()