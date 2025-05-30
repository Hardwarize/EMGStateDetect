import io
import os
import re
from pathlib import Path
import requests
from typing import List, Literal, Annotated, Optional, Any

import numpy as np
import numpy.typing as npt

from pydantic import BaseModel, Field, model_validator, TypeAdapter

import scipy.io as sio

from emgstatedetect.utils.log import get_logger
logger = get_logger(__name__)

from .biopatrec_consts import (
    BIOPATREC_DATA_REPOSITORY_GITHUB_REPO_BASE_URL,
    BIOPATREC_DATA_REPOSITORY_DOWNLOAD_GITHUB_BASE_URL
    )


class BioPatRecField(BaseModel):
    """
    Class representing a field in a BioPatRec recording session.
    Each field has:
    - `code`: A short code representing the field content as is described in BioPatRec repository.
    - `definition`: A short explanation of the field code's meaning.
    - `data_type`: The expected type of the field data.
    - `value`: The final processed data, contains the info for tehe field.
    """
    code: Annotated[
        str,
        Field(description="Short code that represent the field content"),
    ]

    definition: Annotated[
        str,
        Field(description="Short explanation of the field code's meaning"),
    ]

    data_type: Annotated[
        Any,
        Field(description="Expected type of the field data"),
    ]

    value: Annotated[
        Any, Field(description="Final processed data")]

    @model_validator(mode="before")
    @classmethod
    def init_from_raw(cls, data):
        code = data["code"]
        raw_data = data.pop("raw_data", None)

        try:
            # Attempt to convert raw_data to the declared type
            raw_value = raw_data[code].item()

            if isinstance(raw_value, str):
                pass
            
            elif isinstance(raw_value, list):
                if len(raw_value) == 0:
                    # If field exists but is empty...
                    data["value"] = None
                    return data
                else:
                    pass
                
            elif isinstance(raw_value, np.ndarray):
                if raw_value.size == 0:
                    # If field exists but is empty...
                    data["value"] = None
                    return data
                else:
                    pass
                
            elif not np.any(raw_value):
                # If field exists but is empty...
                data["value"] = None
                return data

        except ValueError as ve:
            data["value"] = None
            return data

        dtype = data.get("data_type")

        try:
            parsed_data = TypeAdapter(dtype, config={"arbitrary_types_allowed": True}).validate_python(raw_value)
        except Exception as e:
            raise ValueError(f"Failed to convert raw_data to {dtype}: {e}")

        data["value"] = parsed_data

        return data


class BioPatRecRecordingSession():
    """
    Class representing a BioPatRec recording session.
    A session in BiopatRec contains the following fields:
    - `sample_rate_khz`: Sampling frequency in KHz.
    - `sample_time`: Sampling time.
    - `relaxation_time`: Relaxation time.
    - `movements_count`: Number of movements.
    - `repetitions_count`: Number of repetitions.
    - `channels_count`: Number of channels.
    - `recording_device_name`: Device used for the recordings.
    - `comm_mode`: Communication mode, Wifi or COM.
    - `comm_port_name`: COM port name, available only in case of COM communication.
    - `movements_name`: Description of the movements performed.
    - `date`: Recording date.
    - `comment`: Recording comments.
    - `data`: Signal data array with shape: Samples x Channels x Movements.
    Each field is an instance of `BioPatRecField`.
    """

    def __init__(self, biopatrec_recsession: npt.NDArray):
        
        self.sample_rate_khz: BioPatRecField = BioPatRecField(
            code = "sF",
            definition = "Sampling frequency in KHz",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.sample_time: BioPatRecField = BioPatRecField(
            code = "sT",
            definition = "Sampling time",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.relaxation_time: BioPatRecField = BioPatRecField(
            code = "rT",
            definition = "Relaxation time",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.movements_count: BioPatRecField = BioPatRecField(
            code = "nM",
            definition = "Number of movements",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.repetitions_count: BioPatRecField = BioPatRecField(
            code = "nR",
            definition = "Number of repetitions",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.channels_count: BioPatRecField = BioPatRecField(
            code = "nCh",
            definition = "Number of channels",
            data_type = int,
            raw_data = biopatrec_recsession
            )

        self.recording_device_name: BioPatRecField = BioPatRecField(
            code = "dev",
            definition = "Device used for the recordings",
            data_type = str,
            raw_data = biopatrec_recsession
            )

        self.comm_mode: BioPatRecField = BioPatRecField(
            code = "comm",
            definition = "Communication mode, Wifi or COM",
            data_type = str,
            raw_data = biopatrec_recsession
            )

        self.comm_port_name: BioPatRecField = BioPatRecField(
            code = "comn",
            definition = "COM port name, available only in case of COM communication",
            data_type = str,
            raw_data = biopatrec_recsession
            )

        self.movements_name: BioPatRecField = BioPatRecField(
            code = "mov",
            definition = "Description of the movements performed",
            data_type = List[str],
            raw_data = biopatrec_recsession
            )

        self.date: BioPatRecField = BioPatRecField(
            code = "date",
            definition = "Recording date",
            data_type = npt.NDArray,
            raw_data = biopatrec_recsession
            )

        self.comment: BioPatRecField = BioPatRecField(
            code = "cmt",
            definition = "Recording comments",
            data_type = Optional[str],
            raw_data = biopatrec_recsession
            )

        self.data: BioPatRecField = BioPatRecField(
            code = "tdata",
            definition = "Total data, Samples x Channels x Movements",
            data_type = npt.NDArray,
            raw_data = biopatrec_recsession
            )


class BioPatRecRecordingDataset:
    """
    Class to manage datasets from the BioPatRec repository or a local folder.
    
    This class accept a BioPatRec dataset and a local folder path:
    If the local folder does not exist:
    - It creates the local folder.
    - Downloads the dataset from the BioPatRec GitHub repository.
    - Store the dataset using the local folder as a base path, all the files will be stored in the same folder
    
    If the local folder exists:
    - It checks for the presence of .mat files.
        - If .mat files are found, it loads them.
        - If no .mat files are found, it downloads the dataset from the BioPatRec GitHub repository.
          Notice that givrn that we can not know how many .mat files are in the dataset,
          if just one mat file exists, it is assumed that the dataset is already downloaded.
    
    If no local folder is provided:
        - It downloads the dataset from the BioPatRec GitHub repository.
        - Stores the dataset in the same path as the executing script, all the files will be stored in the same folder.

    Attributes:
        name (str): Name of the dataset.
        info (dict): Dictionary containing recording session information.
    """

    def __init__(self, dataset_name, local_folder=None, persist_source: bool = True):
        """
        Initializes the BioPatRecRecordingDataset instance.

        Args:
            dataset_name (str): The name of the dataset.
            local_folder (str, optional): Path to a local folder with .mat files.
        """
        logger.info(f"Initializing BioPatRecRecordingDataset for: {dataset_name}")
        self.name = dataset_name
        self.local_folder = local_folder
        self.persist_source = persist_source
        
        self.absolute_path_folder = Path.cwd() / self.local_folder if self.local_folder else Path.cwd()

        if self.local_folder:
            print(f"Local folder path: {self.absolute_path_folder}")
            
            if self.absolute_path_folder.is_dir():
                
                mat_files = list(self.absolute_path_folder.rglob('*.mat'))
                
                if mat_files:
                    logger.info(f"Local folder '{self.local_folder}' exists and contains mat files. Loading existing .mat files.")
                    self.info = self.__load_mat_files_from_local()
                    
                else:
                    self.info = self.__get_data_from_source()
                    
            else:
                self.info = self.__get_data_from_source()
                
        else:
            self.info = self.__get_data_from_source()


    def __extract_mat_files_github(self):
        """
        Extracts all MATLAB (.mat) file names from the specified GitHub repository HTML page.

        Returns:
            list: A list of .mat file names extracted from the repository.
        """
        logger.info(f"Fetching .mat file list from GitHub repo URL: {BIOPATREC_DATA_REPOSITORY_GITHUB_REPO_BASE_URL}/{self.name}")
        text = requests.get(f'{BIOPATREC_DATA_REPOSITORY_GITHUB_REPO_BASE_URL}/{self.name}').text
        pattern = r'"([^"]*/[^"]*?\.mat)"'
        recording_sessions_files_names = re.findall(pattern, text)
        logger.info(f"Extracted {len(recording_sessions_files_names)} .mat files from GitHub.")
        return recording_sessions_files_names


    def __download_mat_files_github(self, github_file_path: str):
        url = f"{BIOPATREC_DATA_REPOSITORY_DOWNLOAD_GITHUB_BASE_URL}/{github_file_path}"
        logger.info(f"Downloading file from GitHub: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            file_buffer = io.BytesIO(response.content)
            mat_contents = sio.loadmat(file_buffer, squeeze_me=True)
            return mat_contents
        else:
            logger.error(f"Failed to download file: {url} (status code: {response.status_code})")
            raise FileNotFoundError(f"Failed to download file: {url}")
    
    
    def __get_data_from_source(self):
        dataset_mat_files_names = self.__extract_mat_files_github()
        mat_files_info = [(file, self.__download_mat_files_github(file)) for file in dataset_mat_files_names]
        
        if self.persist_source:
            for file_name, mat_contents in mat_files_info:
                file_local_path = self.absolute_path_folder / file_name
                file_local_path.parent.mkdir(parents=True, exist_ok=True)
                sio.savemat(file_local_path, mat_contents)
        
        info = {
            file_info[0]: BioPatRecRecordingSession(file_info[1]['recSession'])
            for file_info in mat_files_info
        }
        
        return info
    
    
    def __load_mat_files_from_local(self):
        mat_files_path = list(self.absolute_path_folder.rglob('*.mat'))
        mat_files_info = [(mat_file_path.name, sio.loadmat(mat_file_path, squeeze_me=True)) for mat_file_path in mat_files_path]
        info = {
            file_info[0]: BioPatRecRecordingSession(file_info[1]['recSession'])
            for file_info in mat_files_info
        }
        return info
