from libemg.data_handler import OfflineDataHandler

class DataStage:
    
    name: str
    subfolder_path: str
    description: str
    
    def __init__(self, odh: OfflineDataHandler = None):
        self.odh = odh


class RawStage(DataStage):
    name = "Raw"
    subfolder_path = "raw"
    description = "This stage is for raw data. Raw means data as readed from .mat files without any processing."


class FrequencyFilteredStage(DataStage):
    name = "Frequency Filtered"
    subfolder_path = "freq_filtered"
    description = "This stage is for frequency filtered data. Frequency filtering is done to remove noise from the raw signal."


class GoldenDataStage(DataStage):
    name = "Golden Data"
    subfolder_path = "golden_data"
    description = "This stage is for golden data. Golden data is the data that has been manually verified and is considered to be of high quality."\
        "It is used to test classifiers."


class AllKmeansNoRestStage(DataStage):
    name = "All Kmeans No Rest"
    subfolder_path = "all_kmeans_no_rest"
    description = "This stage contains data that results from remove from filtered data the rest state using kmeans method."


class AllPct70NoRestStage(DataStage):
    name = "All Pct 70 No Rest"
    subfolder_path = "all_pct70_no_rest"
    description = "This stage contains data that results from remove from filtered data the rest state using 70% method."


class TrainKmeansNoRestStage(DataStage):
    name = "Train Kmeans No Rest"
    subfolder_path = "train_kmeans_no_rest"
    description = "This stage contains data that results from remove from filtered data the rest state using 70% method."\
        "This stage does not contain the full data, it contains only the filtered data up to 23k samples because the last 13k has been removed for testing purpouses."


class TrainPct70NoRestStage(DataStage):
    name = "Train Pct 70 No Rest"
    subfolder_path = "train_pct70_no_rest"
    description = "This stage contains data that results from remove from filtered data the rest state using 70% method."\
        "This stage does not contain the full data, it contains only the filtered data up to 23k samples because the last 13k has been removed for testing purpouses."
