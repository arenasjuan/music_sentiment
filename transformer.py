import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import pickle
import time
import tarfile
from functools import reduce
from tensorflow.keras.layers import (Input, MultiHeadAttention, Dense, LayerNormalization, 
                                     Add, Dropout, PReLU, GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l1, l2


def extract_tar_gz(file_path, destination):
    try:
        with tarfile.open(file_path, 'r:gz') as file:
            file.extractall(path=destination)
    except Exception as e:
        print(f"Error while extracting {file_path}: {e}")

if not os.path.exists("./features_of_interest") and os.path.exists("./features_of_interest.tar.gz"):
    print("Extracting features_of_interest.tar.gz...")
    extract_tar_gz("./features_of_interest.tar.gz", ".")

print("Reading CSV files...")
csv_files = glob.glob("./features_of_interest/*.csv")
print(f"Found {len(csv_files)} CSV files.")

sample_df = pd.read_csv(csv_files[0])
list_of_all_columns = sample_df.columns.tolist()

# Constants
BATCH_SIZE = 32
BUFFER_SIZE = 108120
EPOCHS = 100 
WEIGHTS_PATH = 'transformer_weights.h5'
TRAIN_DATASET_PATH = "serialized_data/train_dataset"
TEST_DATASET_PATH = "serialized_data/test_dataset"

os.makedirs(TRAIN_DATASET_PATH, exist_ok=True)
os.makedirs(TEST_DATASET_PATH, exist_ok=True)
FEATURE_COLS = [col_name for col_name in list_of_all_columns if col_name not in ["frameTime", "arousal", "valence"]]
TARGET_COLS = ["arousal", "valence"]


def gather_statistics(file_paths):
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, axis=0)
    mean = combined_df[FEATURE_COLS].mean()
    std = combined_df[FEATURE_COLS].std()
    return mean, std

print("Gathering statistics...")
mean, std = gather_statistics(csv_files)
print("Statistics gathered.")


def load_and_preprocess_dataset(file_path, mean, std):
    print(f"Loading and preprocessing {file_path}...")
    df = pd.read_csv(file_path)
    
    for col in FEATURE_COLS:
        df[col] = (df[col] - mean[col]) / std[col]
    
    return df

def dataframe_to_dataset(df):
    feature_dict = dict(zip(FEATURE_COLS, df[FEATURE_COLS].values.T))
    features = tf.stack(list(feature_dict.values()), axis=-1)
    targets = tf.stack([df['arousal'].values, df['valence'].values], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    return dataset

pickle_filename = 'all_datasets.pkl'

# Load preprocessed datasets if the pickle file exists
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        all_dataframes = pickle.load(f)
    print("Loaded all_dataframes from pickle file.")
else:
    print("File containing all_datasets does not exist; now constructing...")
    all_dataframes = [load_and_preprocess_dataset(file, mean, std) for file in csv_files]
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_dataframes, f)
    print("All dataframes loaded, preprocessed, and saved to pickle file.")

# Convert the dtype of each DataFrame to float64 before converting to TensorFlow datasets
all_dataframes = [df.astype('float64') for df in all_dataframes]

# Convert preprocessed DataFrames back into TensorFlow datasets
all_datasets = [dataframe_to_dataset(df) for df in all_dataframes]

def safe_concatenate(ds_list):
    """Attempt to concatenate datasets and return the problematic dataset index if an error occurs."""
    combined = ds_list[0]
    for idx, ds in enumerate(ds_list[1:], start=1):
        try:
            combined = combined.concatenate(ds)
        except Exception as e:
            print(f"Error encountered while concatenating dataset at index {idx}.")
            # You can also print out samples from the problematic dataset to inspect them
            for sample in ds.take(1):
                print(sample)
            raise e
    return combined

print("Concatenating datasets...")
start_time = time.time()
combined_dataset = safe_concatenate(all_datasets)
end_time = time.time()
print(f"Datasets concatenated in {end_time - start_time} seconds.")

print("Checking for existing train and test datasets...")
# Check if the directories themselves exist
if os.path.isdir(TRAIN_DATASET_PATH) and os.path.isdir(TEST_DATASET_PATH):
    print("Loading existing train and test datasets...")
    train_dataset = tf.data.Dataset.load(TRAIN_DATASET_PATH)
    test_dataset = tf.data.Dataset.load(TEST_DATASET_PATH)
    print("Existing datasets loaded.")
else:
    dataset_size = 108120

    split_fraction = 0.8
    train_size = int(split_fraction * dataset_size)

    print(f"Training size: {train_size} rows.")
    print(f"Test size: {dataset_size - train_size} rows.")

    print("Splitting and shuffling datasets...")
    train_dataset = combined_dataset.take(train_size).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = combined_dataset.skip(train_size).batch(BATCH_SIZE)
    print("Datasets split and shuffled.")

    print("Saving train and test datasets...")
    start_time = time.time()  # Begin timing
    train_dataset.save(TRAIN_DATASET_PATH)
    test_dataset.save(TEST_DATASET_PATH)
    end_time = time.time()  # End timing
    print(f"Datasets saved in {end_time - start_time} seconds.")

print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(test_dataset)}")

def get_positional_encoding(seq_len, d_model):
    angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    pos_encoding = np.zeros(angles.shape)
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_block(input_layer, d_model, num_heads, dff, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_layer, input_layer)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([input_layer, attn_output]))
    ff_output = Dense(dff)(out1)
    ff_output = PReLU()(ff_output)
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))
    return out2

def build_transformer_model(input_shape, num_heads, d_model, dff, rate=0.1, num_blocks=6, reg_strength=0.001):
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    pos_enc = get_positional_encoding(input_shape[0], d_model)
    x = Add()([x, pos_enc])
    for _ in range(num_blocks):
        x = transformer_block(x, d_model, num_heads, dff, rate)
    transformer_out = GlobalAveragePooling1D()(x)
    x = Dense(512)(transformer_out)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

d_model = 128
dff = 512
num_heads = 4

sample_features, _ = next(iter(train_dataset))
input_shape = sample_features.shape[1:]

print("Model Configuration:")
print(f"d_model: {d_model}")
print(f"dff: {dff}")
print(f"num_heads: {num_heads}")
print(f"Input Shape: {input_shape}")

checkpoint_dir = "model_checkpoints"

weights_file_path = os.path.join(checkpoint_dir, WEIGHTS_PATH)
print("Building model architecture...")
model = build_transformer_model(input_shape, num_heads, d_model, dff)

if os.path.exists(weights_file_path):
    print("Loading weights from saved checkpoint...")
    try:
        model.load_weights(weights_file_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
else:
    print("No weights found. Initializing model with random weights.")
    os.makedirs(checkpoint_dir, exist_ok=True)

model.compile(optimizer='adam', loss='mse')
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

checkpoint_callback = ModelCheckpoint(filepath=weights_file_path,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      monitor='val_loss',
                                      verbose=1)

callbacks_list = [reduce_lr, checkpoint_callback, early_stopping]

# Training the model
print(f"Starting model training for {EPOCHS} epochs...")
history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
print(f"Final Training Loss: {final_train_loss}")
print(f"Final Validation Loss: {final_val_loss}")