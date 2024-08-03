import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split


data_dir = data_dir = '/Users/omama/Documents/Portfolio/respiratory_classification/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'

# Directory to save spectrograms
spectrogram_dir = '/Users/omama/Documents/Portfolio/respiratory_classification/data'
os.makedirs(spectrogram_dir, exist_ok=True)

# Function to generate and save spectrogram
def generate_spectrogram(wav_path, save_path):
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, spectrogram_data = spectrogram(samples, sample_rate)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.log(spectrogram_data + 1e-10), aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#Generate spectrograms for all audio files
audio_files = glob.glob(os.path.join(data_dir, '*.wav'))

for audio_file in audio_files:
    filename = os.path.basename(audio_file).replace('.wav', '.png')
    save_path = os.path.join(spectrogram_dir, filename)
    generate_spectrogram(audio_file, save_path)
    
def load_and_process_data():
    # Load merged dataframe
    merged_df = pd.read_csv('/Users/omama/Documents/Portfolio/respiratory_classification/data/processed_respiratory_data.csv')

    # Create a dictionary mapping filenames to diagnosis labels
    label_dict = dict(zip(merged_df['filename'].str.replace('.wav', '.png'), merged_df['Diagnosis']))

    # Sort the unique labels and map them to numerical values for consistency
    sorted_labels = sorted(set(label_dict.values()))
    label_mapping = {label: idx + 1 for idx, label in enumerate(sorted_labels)}

    # Print the label mapping
    print("Label Mapping:")
    for label, num in label_mapping.items():
        print(f"{label} -> {num}")
        
    # Convert each label in label_dict to its numerical value
    numerical_labels = {k: label_mapping[v] for k, v in label_dict.items()}

    # Split the data into filenames and corresponding numerical labels
    filenames = list(numerical_labels.keys())
    labels = list(numerical_labels.values())
    
    return filenames, labels

# if __name__ == "__main__":
#     filenames, labels = load_and_process_data()
#     print(filenames[:30], labels[:30])

# DEBUGGING
# Print the unique mappings
# print("\nUnique Mappings:")
# unique_mappings = {label_mapping[v]: v for v in set(label_dict.values())}
# for num, label in unique_mappings.items():
#     print(f"{num} -> {label}")

# file_path = '/Users/omama/Documents/Portfolio/respiratory_classification/data/processed_respiratory_data.csv'

# if os.path.exists(file_path):
#     print("File exists.")
# else:
#     print("File does not exist. Please check the path.")