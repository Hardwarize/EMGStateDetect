# streamlit run streamlit.py

from copy import deepcopy
from libemg.filtering import Filter
from libemg.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.cluster import KMeans
import itertools
import os
from matplotlib import pyplot as plt

from emgstatedetect.providers.biopatrec.biopatrec_classes import BioPatRecRecordingDataset
from emgstatedetect.providers.biopatrec.biopatrec_libemg_adapters.adapter_10mov4chFU_AFEs import Device, AnalogFrontEnd_UntargetedForearm
from emgstatedetect.visualizations.plots import plot_raw_segmented_clean_one_subject_one_class
from emgstatedetect.state_detection.class_asignment import map_window_labels_to_samples_by_voting, assign_class_to_labels

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle



fe = FeatureExtractor()

@st.cache_resource
def get_database_session(url):
    dataset = BioPatRecRecordingDataset('10mov4chFU_AFEs', '10mov4chFU_AFEs', persist_source=True)

    TIADS1299_dataset = AnalogFrontEnd_UntargetedForearm(dataset_info=dataset.info, device=Device.ADS)

    TIADS1299_dataset.prepare_data()
    TIADS1299_dataset.filter_data()

    WINDOW_SECONDS_SIZE = 0.25
    WINDOW_SECONDS_STEP = 0.05

    window_samples_size = int(TIADS1299_dataset.sampling * WINDOW_SECONDS_SIZE)
    window_samples_step = int(TIADS1299_dataset.sampling * WINDOW_SECONDS_STEP)
    
    unique_subjects = list(np.unique(np.array(TIADS1299_dataset.odh.subjects)))
    unique_classes = list(np.unique(np.array(TIADS1299_dataset.odh.classes)))

    combinations = list(itertools.product(unique_subjects, unique_classes))

    clean_signal = {}
    
    for subject, class_ in combinations:
        print(f"Processing subject {subject}, class {class_}")
        odh = TIADS1299_dataset.odh.isolate_data("subjects", [subject])
        odh = odh.isolate_data("classes", [class_])
            
        windows, metadata = odh.parse_windows(window_samples_size, window_samples_step, metadata_operations=None)

        features = fe.extract_feature_group('HTD', windows, array=True)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')

        # Fit and predict cluster labels
        labels = kmeans.fit_predict(features)

        samples_label = map_window_labels_to_samples_by_voting(labels, window_samples_size, window_samples_step, odh.data[0].shape[0])

        classes_dict = assign_class_to_labels(odh.data[0], samples_label)
        
        #classes_dict['Action']['data'] = classes_dict['Action']['data'][np.any(np.abs(classes_dict['Action']['data']) > 0.0001, axis=1)]
        
        #fig = plot_raw_segmented_clean_one_subject_one_class(odh.data[0], odh.data[0], classes_dict['Action']['data'], classes_dict, samples_label)
        """
        if clean_signal.get(subject):
            clean_signal[subject][class_] = fig
        else:
            clean_signal[subject] = {class_: fig}
        """
        # Save the figure to a pickle file
        pickle_filename = f"/workspaces/EMGStateDetect/UI_outputs/subject_{subject}_class_{class_}_figure.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump({'data': odh.data[0], 'val': classes_dict, 'labs':samples_label}, f)
        plt.close()
        
        
    return clean_signal


#sc = get_database_session("hi")


# Dropdown for channel selection
subject = st.selectbox(
    "Select Subject to plot",
    options=[0, 1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: f"Subject {x}"
)

mov = st.selectbox(
    "Select Mov to plot",
    options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    format_func=lambda x: f"Mov {x}"
)


# Prepare data for plotting
#fig2 = sc[subject][mov]

# Optionally load the figure from pickle file
pickle_filename = f"/workspaces/EMGStateDetect/UI_outputs/subject_{subject}_class_{mov}_figure.pkl"
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        dct = pickle.load(f)

fig2 = plot_raw_segmented_clean_one_subject_one_class(dct['data'], dct['data'], dct['val']['Action']['data'], dct['val'], dct['labs'])
# Create two columns
st.pyplot(fig2)