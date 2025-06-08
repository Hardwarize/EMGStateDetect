from copy import deepcopy
from enum import Enum
import itertools
import os
from pathlib import Path
import re
from typing import List

from libemg._datasets.dataset import Dataset
from libemg.filtering import Filter
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor

import numpy as np

from emgstatedetect.state_detection.methods import detector_kmeans, fixed_time_70_pct
from emgstatedetect.state_detection.class_asignment import map_window_labels_to_samples_by_voting, assign_class_to_labels

from emgstatedetect.data_stages.stages import *

class Device(Enum):
    ADS = "ADS"
    ADSbias = "ADSbias"
    INTAN = "INTAN"


class ValuesState(Enum):
    RAW = "raw"
    CLEAN = "clean"
    BASELINE = "baseline"


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
        self.persist_raw_data_to_libemg_structure(dataset_info)
        
        # Data stages
        self.raw_stage = RawStage()
        self.freq_filtered_stage = FrequencyFilteredStage()
        self.golden_data_stage = GoldenDataStage()
        self.all_kmeans_no_rest_stage = AllKmeansNoRestStage()
        self.all_pct70_no_rest_stage = AllPct70NoRestStage()
        self.train_kmeans_no_rest_stage = TrainKmeansNoRestStage()
        self.train_pct70_no_rest_stage = TrainPct70NoRestStage()

        self.stages_lst = [
            self.raw_stage,
            self.freq_filtered_stage,
            self.golden_data_stage,
            self.all_kmeans_no_rest_stage,
            self.all_pct70_no_rest_stage,
            self.train_kmeans_no_rest_stage,
            self.train_pct70_no_rest_stage
        ]
    
    
    def get_stage(self, stage_name: str) -> DataStage:
        """
        Returns the stage with the given name.
        """
        for stage in self.stages_lst:
            if stage.name == stage_name:
                return stage
        raise ValueError(f"Stage {stage_name} not found in dataset {self.device_name}. Available stages: {[s.name for s in self.stages_lst]}")
    
    
    def persist_raw_data_to_libemg_structure(self, dataset_info):
    
        # Filter dataset_info to only include entries for the specified device_name
        device_ds_info = {k: v for k, v in dataset_info.items() if re.search(r'\d+_(.*?)\.mat', k).group(1) == self.device_name}

        for session_name, session_values in device_ds_info.items():
            for gesture_id, _ in enumerate(session_values.movements_name.value):

                values_matrix = session_values.data.value[:,:,gesture_id]
                subject_id = session_name.split('_')[0]
                
                # We need to subtract 1 from subject_id to match the libemg structure (0-indexed)
                file_path = Path(self.device_name) / RawStage.subfolder_path / f"S_{int(subject_id)-1}_C_{gesture_id}.csv"
                
                full_path = Path.cwd() / DATASET_FOLDER / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # The comments param is required, if not provided, it will default to '#'
                np.savetxt(full_path, values_matrix, delimiter=",", header=','.join([f'ch{i}' for i in range(self.num_channels)]), comments='')
        

    def load_data(self, data_stage: DataStage):
        """
        This function uses libemg OfflineDataHandler class to load the dataset from
        a local folder.
        """
        subject_list = np.arange(NUM_SUBJECTS)
        subjects_values = [str(s) for s in subject_list]

        classes_values = [str(i) for i in range(len(GESTURES.items()))]

        regex_filters = [
            RegexFilter(left_bound = "/S_", right_bound="_", values = subjects_values, description='subjects'),
            RegexFilter(left_bound = "C_", right_bound=".csv", values = classes_values, description='classes')
        ]
        
        odh = OfflineDataHandler()
        odh.get_data(folder_location = Path(DATASET_FOLDER) / self.device_name / data_stage.subfolder_path, regex_filters=regex_filters, delimiter=",", skiprows=1)
        
        stage = self.get_stage(data_stage.name)
        stage.odh = deepcopy(odh)


    def reset_data(self):
        self.prepare_data()
    
    
    def get_extra_attributes_combinations(self):
        unique_subjects = list(np.unique(np.array(self.raw_stage.odh.subjects)))
        unique_classes = list(np.unique(np.array(self.raw_stage.odh.classes)))
    
        combinations = list(itertools.product(unique_subjects, unique_classes))
        return combinations

    
    def get_frequency_filtered_data(self):
        tmp_odh = deepcopy(self.raw_stage.odh)
        filter = Filter(self.sampling)
        filter_configs = [
            {'name': 'highpass', 'cutoff': 20, 'order': 4},      # Remove low frequency noise (<20 Hz)
            {'name': 'notch', 'cutoff': 50, 'bandwidth': 1},     # Remove power network (50 Hz)
            {'name': 'lowpass', 'cutoff': 450, 'order': 4}       # Remove high frequency noise (>450 Hz)
        ]
        
        for filter_config in filter_configs:
            filter.install_filters(filter_config)
            
        filter.filter(tmp_odh)
        self.freq_filtered_stage.odh = tmp_odh


    def get_features(self, windows):
        fe = FeatureExtractor()
        features = fe.extract_feature_group('HTD', windows, array=True)
        return features
    
    
    def persist_data(self, data_stage: DataStage):
        stage = self.get_stage(data_stage.name)
        
        for subject, class_ in self.get_extra_attributes_combinations():
            tmp_odh = stage.odh.isolate_data("subjects", [subject])
            isolated_odh = tmp_odh.isolate_data("classes", [class_])
            os.makedirs(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{stage.subfolder_path}/', exist_ok=True)
            np.savetxt(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{stage.subfolder_path}/S_{subject}_C_{class_}.csv', isolated_odh.data[0], delimiter=',', header=','.join([f'ch{i}' for i in range(self.num_channels)]), comments='')


    def clean_data(self, destination_stage: DataStage):
        
        allowed_stages = [AllKmeansNoRestStage.name, AllPct70NoRestStage.name, TrainKmeansNoRestStage.name, TrainPct70NoRestStage.name]
        assert destination_stage.name in allowed_stages, f"destination_stage must be an instance of any: {allowed_stages}"
        
        if destination_stage.name in [AllKmeansNoRestStage.name, TrainKmeansNoRestStage.name]:
            self._clean_data_kmeans(destination_stage)
        elif destination_stage.name in [AllPct70NoRestStage.name, TrainPct70NoRestStage.name]:
            self._clean_data_pct70(destination_stage)


    def _clean_data_pct70(self, destination_stage: DataStage):
        allowed_stages = (TrainPct70NoRestStage.name, AllPct70NoRestStage.name)
        assert destination_stage.name in allowed_stages, f"destination_stage must be an instance of any: {allowed_stages}"
            
        for subject, class_ in self.get_extra_attributes_combinations():
            isolate_odh = self.freq_filtered_stage.odh.isolate_data("subjects", [subject]).isolate_data("classes", [class_])
    
            if destination_stage.name == TrainPct70NoRestStage.name:
                data_to_clean = isolate_odh.data[0][:23000, :]
            elif  destination_stage.name == AllPct70NoRestStage.name:
                data_to_clean = isolate_odh.data[0]
            
            data = fixed_time_70_pct(self.sampling, data_to_clean)
            os.makedirs(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{destination_stage.subfolder_path}/', exist_ok=True)
            np.savetxt(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{destination_stage.subfolder_path}/S_{subject}_C_{class_}.csv', data, delimiter=',')
        

    def _clean_data_kmeans(self, destination_stage: DataStage):
        allowed_stages = [TrainKmeansNoRestStage.name, AllKmeansNoRestStage.name]
        assert destination_stage.name in allowed_stages, f"destination_stage must be an instance of any: {allowed_stages}"
        
        window_seconds_size = 0.25
        window_seconds_step = 0.05
        window_samples_size = int(window_seconds_size*self.sampling)
        window_samples_step = int(window_seconds_step*self.sampling)
        
        for subject, class_ in self.get_extra_attributes_combinations():
            
            isolate_odh = self.freq_filtered_stage.odh.isolate_data("subjects", [subject]).isolate_data("classes", [class_])
            print(f"Processing subject {subject}, class {class_}, {isolate_odh.data[0].shape}")
            
            if destination_stage.name == TrainKmeansNoRestStage.name:
                isolate_odh.data[0] = isolate_odh.data[0][:23000, :]
            
            windows, _ = isolate_odh.parse_windows(int(window_seconds_size*self.sampling), int(window_seconds_step*self.sampling))
            features = self.get_features(windows)
            labels, centroids = detector_kmeans(features)
            samples_label = map_window_labels_to_samples_by_voting(labels, window_samples_size, window_samples_step, isolate_odh.data[0].shape[0])
            classes_dict = assign_class_to_labels(isolate_odh.data[0], samples_label)

            os.makedirs(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{destination_stage.subfolder_path}/', exist_ok=True)
            np.savetxt(f'/workspaces/EMGStateDetect/10mov4chFU_AFEs/{self.device_name}/{destination_stage.subfolder_path}/S_{subject}_C_{class_}.csv', classes_dict['Action']['data'], delimiter=',')
        

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
