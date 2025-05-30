from enum import Enum
import os
from pathlib import Path
import re

from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter

import numpy as np

class Device(Enum):
    ADS = "ADS"
    ADSbias = "ADSbias"
    INTAN = "INTAN"


# Commons constants for the datasets
SAMPLING_FREQUENCY = 2000
NUM_CHANNELS = 4
NUM_SUBJECTS = 8
STUDY_URL = 'https://ieeexplore.ieee.org/document/7318805?arnumber=7318805'
STUDY_TITLE = 'Analog front-ends comparison in the way of a portable, low-power and low-cost EMG controller based on pattern recognition'
STUDY_NUM_REPS = 'Two per subject'
GESTURES = {0: 'Open Hand', 1: 'Close Hand', 2: 'Flex Hand',
            3: 'Extend Hand', 4: 'Pronation', 5: 'Supination',
            6: 'Side Grip', 7: 'Fine Grip', 8: 'Agree', 9: 'Pointer'}
DATASET_FOLDER = "10mov4chFU_AFEs"
DATASET_ROOT_PATH=""

DEVICE_MAP = {
    Device.ADS.value: 'TI ADS1299',
    Device.ADSbias.value: 'TI ADS1299 with bias-driver enabled',
    Device.INTAN.value: 'Intan RHA2216'
    }


class AnalogFrontEnd_UntargetedForearm(Dataset):
    def __init__(self, dataset_info = None, dataset_folder=None, device: Device = None):
        assert dataset_info is not None, "dataset_info must be provided to persist data to libemg structure"
        assert device is not None, "device must be provided to persist data to libemg structure"
        assert isinstance(device, Device), "device must be an instance of Device Enum"
        
        Dataset.__init__(self,
                          SAMPLING_FREQUENCY,
                          NUM_CHANNELS,
                          DEVICE_MAP[device.value],
                          NUM_SUBJECTS,
                          GESTURES,
                          STUDY_NUM_REPS,
                          STUDY_TITLE,
                          STUDY_URL)

        self.dataset_folder = dataset_folder
        self.device_name = device.value
        self.persist_data_to_libemg_structure(dataset_info)
    
    
    def persist_data_to_libemg_structure(self, dataset_info):
    
        # Filter dataset_info to only include entries for the specified device_name
        device_ds_info = {k: v for k, v in dataset_info.items() if re.search(r'\d+_(.*?)\.mat', k).group(1) == self.device_name}

        for session_name, session_values in device_ds_info.items():
            for gesture_id, _ in enumerate(session_values.movements_name.value):

                values_matrix = session_values.data.value[:,:,gesture_id]
                subject_id = session_name.split('_')[0]

                file_path = Path(self.device_name) / f"Subject_{subject_id}" /  f"C_{gesture_id}.csv"
                
                full_path = Path.cwd() / DATASET_FOLDER / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                np.savetxt(full_path, values_matrix, delimiter=",")
        

    def prepare_data(self):
        """
        This function uses libemg OfflineDataHandler class to load the dataset from
        a local folder.
        """
        subject_list = np.array(list(range(1, NUM_SUBJECTS+1)))
        subjects_values = [str(s) for s in subject_list]

        classes_values = [str(i) for i in range(len(GESTURES.items()))]

        regex_filters = [
            RegexFilter(left_bound = "/Subject_", right_bound="/", values = subjects_values, description='subjects'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        
        odh = OfflineDataHandler()
        odh.get_data(folder_location = Path(DATASET_FOLDER) / self.device_name, regex_filters=regex_filters, delimiter=",")

        return odh

"""
class AnalogFrontEnd_UntargetedForearm_TIADS1299(Dataset):
    def __init__(self, dataset_info = None, dataset_folder=None, device_folder='ADS'):
        assert dataset_info is not None, "dataset_info must be provided to persist data to libemg structure"
        
        Dataset.__init__(self,
                          SAMPLING_FREQUENCY,
                          NUM_CHANNELS,
                          'TI ADS1299',
                          NUM_SUBJECTS,
                          GESTURES,
                          STUDY_NUM_REPS,
                          STUDY_TITLE,
                          STUDY_URL)

        self.dataset_folder = dataset_folder
        self.device_folder = device_folder
        persist_data_to_libemg_structure(dataset_info, self.device_folder)


        

    def prepare_data(self):
        return prepare_data_AnalogFrontEnd_UntargetedForearm(self.dataset_folder, self.device_folder)


class AnalogFrontEnd_UntargetedForearm_TIADS1299BiasDriver(Dataset):
    def __init__(self, dataset_folder=DATASET_ROOT_PATH, device_folder='ADSbias'):
        Dataset.__init__(self,
                        SAMPLING_FREQUENCY,
                        NUM_CHANNELS,
                        'TI ADS1299 with bias-driver enabled',
                        NUM_SUBJECTS,
                        GESTURES,
                        STUDY_NUM_REPS,
                        STUDY_TITLE,
                        STUDY_URL)

        self.dataset_folder = dataset_folder
        self.device_folder = device_folder


    def prepare_data(self):
        return prepare_data_AnalogFrontEnd_UntargetedForearm(self.dataset_folder, self.device_folder)


class AnalogFrontEnd_UntargetedForearm_IntanRHA2216(Dataset):
    def __init__(self, dataset_folder=DATASET_ROOT_PATH, device_folder='INTAN'):
        Dataset.__init__(self,
                        SAMPLING_FREQUENCY,
                        NUM_CHANNELS,
                        'Intan RHA2216',
                        NUM_SUBJECTS,
                        GESTURES,
                        STUDY_NUM_REPS,
                        STUDY_TITLE,
                        STUDY_URL)

        self.dataset_folder = dataset_folder
        self.device_folder = device_folder


    def prepare_data(self):
        return prepare_data_AnalogFrontEnd_UntargetedForearm(self.dataset_folder, self.device_folder)
"""