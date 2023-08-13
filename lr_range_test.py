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
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


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
BATCH_SIZE = 10
EPOCHS = 100
WEIGHTS_PATH = 'transformer_weights.h5'
TRAIN_DATASET_PATH = f"serialized_data/train_dataset_batch_{BATCH_SIZE}"
TEST_DATASET_PATH = f"serialized_data/test_dataset_batch_{BATCH_SIZE}"
os.makedirs(TRAIN_DATASET_PATH, exist_ok=True)
os.makedirs(TEST_DATASET_PATH, exist_ok=True)
FEATURE_COLS = [col_name for col_name in list_of_all_columns if col_name not in ["frameTime", "arousal", "valence"]]
TARGET_COLS = ["arousal", "valence"]

def load_datasets(file_paths):
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, axis=0)
    return combined_df.copy()

def gather_statistics(df):
    mean = df[FEATURE_COLS].mean()
    std = df[FEATURE_COLS].std()
    return mean, std

def load_and_preprocess_dataset(df, mean, std):
    for col in FEATURE_COLS:
        df.loc[:, col] = (df[col] - mean[col]) / std[col]
    return df

def dataframe_to_dataset(df):
    feature_dict = dict(zip(FEATURE_COLS, df[FEATURE_COLS].values.T))
    features = tf.stack(list(feature_dict.values()), axis=-1)
    targets = tf.stack([df['arousal'].values, df['valence'].values], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    return dataset

pickle_filename = 'all_datasets.pkl'
combined_df = load_datasets(csv_files)

dataset_size = len(combined_df)
BUFFER_SIZE = dataset_size
split_fraction = 0.8
train_size = int(split_fraction * dataset_size)

# Split the datasets into training and test first
train_df = combined_df.iloc[:train_size].copy()
test_df = combined_df.iloc[train_size:].copy()

# Gather statistics only on the training set
print("Gathering statistics...")
mean, std = gather_statistics(train_df)
print("Statistics gathered.")

if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        all_dataframes = pickle.load(f)
    print("Loaded all_dataframes from pickle file.")
else:
    all_dataframes = [load_and_preprocess_dataset(df_chunk, mean, std) for df_chunk in [train_df, test_df]]
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_dataframes, f)
    print("All dataframes loaded, preprocessed, and saved to pickle file.")

all_dataframes = [df.astype('float64') for df in all_dataframes]
all_datasets = [dataframe_to_dataset(df) for df in all_dataframes]

def safe_concatenate(ds_list):
    combined = ds_list[0]
    for idx, ds in enumerate(ds_list[1:], start=1):
        try:
            combined = combined.concatenate(ds)
        except Exception as e:
            print(f"Error encountered while concatenating dataset at index {idx}.")
            for sample in ds.take(1):
                print(sample)
            raise e
    return combined

train_dataset, test_dataset = all_datasets

print("Checking for existing train and test datasets...")
if os.path.exists(os.path.join(TRAIN_DATASET_PATH, "dataset_spec.pb")) and os.path.exists(os.path.join(TEST_DATASET_PATH, "dataset_spec.pb")):
    print("Loading existing train and test datasets...")
    train_dataset = tf.data.Dataset.load(TRAIN_DATASET_PATH)
    test_dataset = tf.data.Dataset.load(TEST_DATASET_PATH)
    print("Existing datasets loaded.")
else:
    print("Splitting and shuffling datasets...")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    print("Datasets split and shuffled.")
    print("Saving train and test datasets...")
    start_time = time.time()
    train_dataset.save(TRAIN_DATASET_PATH)
    test_dataset.save(TEST_DATASET_PATH)
    end_time = time.time()
    print(f"Datasets saved in {end_time - start_time} seconds.")


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

opt = Adam(learning_rate=0.01)  # This will be updated after the learning rate test
model.compile(optimizer=opt, loss='mse')
model.summary()

# [LR Range Test Code]

def create_batches(train_dataset, total_batches, batches_per_epoch):
    all_batches = []
    remaining_batches = total_batches

    while remaining_batches > 0:
        # Shuffle the dataset and take min(remaining_batches, batches_per_epoch) batches
        for batch in train_dataset.shuffle(buffer_size=BUFFER_SIZE).take(min(remaining_batches, batches_per_epoch)):
            all_batches.append(batch)
        
        remaining_batches -= batches_per_epoch

    return iter(all_batches)  # Return an iterator

def find_lr(model, all_batches, test_dataset, min_lr=1e-10, max_lr=0.01, steps=20000, eval_interval=1000):
    current_lr = min_lr
    lr_multiplier = np.exp(np.log(max_lr / min_lr) / steps)
    train_losses = []
    val_losses = []
    lrs = []

    for step_counter, (inputs, targets) in enumerate(all_batches):
        # Early stopping check
        if step_counter >= steps:
            break

        # Set learning rate
        K.set_value(model.optimizer.lr, current_lr)

        # Train one batch and get the loss
        train_loss = model.train_on_batch(inputs, targets)
        train_losses.append(train_loss)  # Append train loss

        # Evaluate the model on the validation set every eval_interval batches
        if step_counter % eval_interval == 0:
            val_loss = model.evaluate(test_dataset, verbose=0)

            # Print progress
            print(f"Step {step_counter}, LR: {current_lr:.10e}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

            # Store the validation loss and learning rate
            val_losses.append(val_loss)
            lrs.append(current_lr)

        # Update the learning rate for the next step
        current_lr *= lr_multiplier

    return lrs, val_losses

# Get the batches
all_batches = create_batches(train_dataset, total_batches=200000, batches_per_epoch=10244)

# Run the LR range test
lrs, val_losses = find_lr(model, all_batches, test_dataset)

# Plot the results
plt.plot(lrs, val_losses)  # Update this to plot val_losses
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')

# Add gridlines
plt.grid(True, which="both", ls="--", c='0.7')

plt.savefig('lr_range_test_plot.png')

min_loss_idx = np.argmin(val_losses)  # Update this to use val_losses
best_lr = lrs[min_loss_idx]
best_val_loss = val_losses[min_loss_idx]
print(f"Best Learning Rate: {best_lr}, Corresponding Validation Loss: {best_val_loss}")