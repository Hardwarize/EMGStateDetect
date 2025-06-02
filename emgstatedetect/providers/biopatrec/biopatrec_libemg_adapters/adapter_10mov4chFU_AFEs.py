from copy import deepcopy
from enum import Enum
import os
from pathlib import Path
import re
from typing import List

from libemg._datasets.dataset import Dataset
from libemg.filtering import Filter
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
        self.raw_odh = None
        self.clean_odh = None
    
    
    def persist_data_to_libemg_structure(self, dataset_info):
    
        # Filter dataset_info to only include entries for the specified device_name
        device_ds_info = {k: v for k, v in dataset_info.items() if re.search(r'\d+_(.*?)\.mat', k).group(1) == self.device_name}

        for session_name, session_values in device_ds_info.items():
            for gesture_id, _ in enumerate(session_values.movements_name.value):

                values_matrix = session_values.data.value[:,:,gesture_id]
                subject_id = session_name.split('_')[0]
                
                # We need to subtract 1 from subject_id to match the libemg structure (0-indexed)
                file_path = Path(self.device_name) / "raw" / f"Subject_{int(subject_id)-1}" /  f"C_{gesture_id}.csv"
                
                full_path = Path.cwd() / DATASET_FOLDER / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                np.savetxt(full_path, values_matrix, delimiter=",")
        

    def prepare_data(self, is_clean: bool = False):
        """
        This function uses libemg OfflineDataHandler class to load the dataset from
        a local folder.
        """
        subject_list = np.arange(NUM_SUBJECTS)
        subjects_values = [str(s) for s in subject_list]

        classes_values = [str(i) for i in range(len(GESTURES.items()))]

        regex_filters = [
            RegexFilter(left_bound = "/Subject_", right_bound="/", values = subjects_values, description='subjects'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        
        odh = OfflineDataHandler()
        odh.get_data(folder_location = Path(DATASET_FOLDER) / self.device_name / ('clean' if is_clean else 'raw'), regex_filters=regex_filters, delimiter=",")

        if is_clean:
            self.clean_odh = deepcopy(odh)
        else:
            self.raw_odh = deepcopy(odh)


    def reset_data(self):
        self.prepare_data()

    
    def filter_data(self):
        odh = deepcopy(self.raw_odh)
        filter = Filter(self.sampling)
        filter.install_common_filters()
        filter.filter(odh)
        self.odh = odh


    def isolate_data(self, subjects: List[int] = None, classes: List[int] = None):
        """
        Isolate data for a specific subject and class.
        """
        if subjects is None and classes is None:
            raise ValueError("At least one of subjects or classes must be provided to isolate data.")

        if subject_id is not None:
            self.odh.filter_by_subject(subject_id)

        if class_id is not None:
            self.odh.filter_by_class(class_id)
