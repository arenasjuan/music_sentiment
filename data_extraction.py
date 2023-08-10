import os
import pandas as pd

FEATURES_DIR = "./features"
ANNOTATIONS_AROUSAL = "./annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
ANNOTATIONS_VALENCE = "./annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv"
DESTINATION_DIR = "./features_of_interest"

df_arousal = pd.read_csv(ANNOTATIONS_AROUSAL, index_col=0)
df_valence = pd.read_csv(ANNOTATIONS_VALENCE, index_col=0)

base_cols_of_interest = [
    "F0final_sma_stddev", "F0final_sma_amean",
    "pcm_RMSenergy_sma_stddev", "pcm_RMSenergy_sma_amean",
    "pcm_zcr_sma_stddev", "pcm_zcr_sma_amean",
    "pcm_fftMag_spectralFlux_sma_stddev", "pcm_fftMag_spectralFlux_sma_amean",
    "pcm_fftMag_spectralCentroid_sma_stddev", "pcm_fftMag_spectralCentroid_sma_amean",
    "pcm_fftMag_spectralRollOff25.0_sma_stddev", "pcm_fftMag_spectralRollOff25.0_sma_amean"
]

time_limits = [f"sample_{ms}ms" for ms in range(15000, 45000, 500)]

for filename in os.listdir(FEATURES_DIR):
    if filename.endswith(".csv"):
        feature_path = os.path.join(FEATURES_DIR, filename)
        song_index = int(filename.split('.')[0])
        
        # Read the CSV header to use as column names later
        with open(feature_path, 'r') as file:
            header_line = file.readline().strip()
            header = header_line.split(';')

        # Read the CSV skipping the initial rows and the header
        df_features = pd.read_csv(feature_path, delimiter=';', header=None, skiprows=31)

        # Rename columns based on the extracted header
        df_features.columns = header
        
        # Convert the DataFrame to float type
        df_features = df_features.astype(float)

        # Reset column names (removing the first one which was just the row with data names)
        cols_of_interest = base_cols_of_interest[:]
        
        # Dynamic MFCC column extraction
        mfcc_cols = [col for col in df_features.columns if "pcm_fftMag_mfcc_sma[" in col and "_stddev" in col]
        for col in mfcc_cols:
            num = col.split("[")[-1].split("]")[0]
            cols_of_interest.extend([f"pcm_fftMag_mfcc_sma[{num}]_stddev", f"pcm_fftMag_mfcc_sma[{num}]_amean"])

        # Filter columns that are in both DataFrame and cols_of_interest
        columns_to_extract = ['frameTime'] + list(set(df_features.columns).intersection(set(cols_of_interest)))
        df_features = df_features[columns_to_extract]

        # Add arousal and valence columns
        df_features['arousal'] = df_features['frameTime'].apply(lambda x: df_arousal.at[song_index, f"sample_{int(float(x) * 1000)}ms"] if f"sample_{int(float(x) * 1000)}ms" in df_arousal.columns else None)
        df_features['valence'] = df_features['frameTime'].apply(lambda x: df_valence.at[song_index, f"sample_{int(float(x) * 1000)}ms"] if f"sample_{int(float(x) * 1000)}ms" in df_valence.columns else None)

        # Filter rows where arousal and valence are not None
        df_features = df_features.dropna(subset=['arousal', 'valence'])

        # Save to destination directory
        if not df_features.empty:
            dest_path = os.path.join(DESTINATION_DIR, filename)
            df_features.to_csv(dest_path, index=False)



print("Data extraction and merging completed!")