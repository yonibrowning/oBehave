from __future__ import print_function
import os
import numpy as np
import pandas as pd

def load_manifest(drive_path = '/data/dynamic-brain-workshop/visual_behavior',
                  manifest_file = 'visual_behavior_data_manifest.csv'):
    '''
    This is just a rapper to load behavior dataset manifest. 
    Inputs 
    drive_path (optional): Location of data on drive. Default is AWS location
    '''
    manifest = pd.read_csv(os.path.join(drive_path,manifest_file))