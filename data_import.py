import time
import os

import numpy as np
import pandas as pd

# The files from the raw data set contain multi index headers.
# These were combined in excel manually to simplify the import.
# For the tracks, we removed columns not realting to the genre information.
# Additionally, the data in the genres has >1000 genres, that went through
# 2 rounds of parent selection so that more "generic" genres were chosen.
# This was done to reduce the total number of genres considered.


tracks = pd.read_csv('fma_metadata/genre_filter_clean.csv', header=0)
features = pd.read_csv('fma_metadata/features.csv')


# export_tracks.to_csv('tracks_genres.csv', index= False)


# print(tracks)
# print(tracks.shape)

tracks.set_index('track_id')

# print(tracks)
# print(features.shape)

# Join the tracks(containing genre information) to the features. 
# Only keep track/genre combos where there are corresponding features.

joined = features.set_index('track_id').join(tracks.set_index('track_id'))

# print(joined)
# print(joined.shape)

# Print the file.
joined.to_csv('cleaned_dataset.csv')


# print(tracks)
# print(features)