## This is the official script in the end for generating the new joint audios

import json
import os
import random
from math import log10  # Import log10 function
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm


def calculate_loudness(audio_segment):
    """
    Calculate the loudness of an audio segment based on its RMS amplitude.
    """
    return audio_segment.rms


# Calculate the necessary adjustment in dB for each audio file to match the reference
def adjustment_dB(target_rms, current_rms):
    return 20 * log10(target_rms / current_rms) if current_rms > 0 else 0


def time_shift_audio(audio_data: np.ndarray, sample_rate: int):
    # Generate a random time shift value in the range [-0.005, 0.005] seconds
    time_shift_sec = random.uniform(-0.1, 0.1)

    # Convert time shift to sample shift
    sample_shift = int(time_shift_sec * sample_rate)

    # Apply the sample shift
    if sample_shift > 0:
        # Positive shift (shift forward): Add zeros at the beginning
        shifted_audio_data = np.pad(audio_data, (sample_shift, 0), mode="constant")[
            :-sample_shift
        ]
    else:
        # Negative shift (shift backward): Add zeros at the end
        shifted_audio_data = np.pad(audio_data, (0, -sample_shift), mode="constant")[
            -sample_shift:
        ]
    return shifted_audio_data


def adjust_volume_librosa(audio_data: np.ndarray, sample_rate: int):
    # Generate a random volume change value in the range [-3, 3] dB
    volume_change_dB = random.uniform(-3, 3)

    # Calculate the linear gain factor from the dB change
    gain_factor = 10 ** (volume_change_dB / 20)

    # Apply the gain factor to adjust volume
    adjusted_audio_data = audio_data * gain_factor
    return adjusted_audio_data


def pitch_shift_librosa(audio_data: np.ndarray, sample_rate: int):
    # Generate a random pitch shift value in the range [-2, 2] semitones
    semitone_shift = random.uniform(-2, 2)

    # Apply the pitch shift
    shifted_audio_data = librosa.effects.pitch_shift(
        y=audio_data, sr=sample_rate, n_steps=semitone_shift
    )
    return shifted_audio_data


def time_stretch_librosa(audio_data: np.ndarray, sample_rate: int):
    # Generate a random time stretch factor in the range [0.8, 1.2]
    time_stretch_factor = random.uniform(0.8, 1.2)

    # Apply time stretching
    stretched_audio_data = librosa.effects.time_stretch(
        y=audio_data, rate=time_stretch_factor
    )

    return stretched_audio_data


def return_identical(audio_data: np.ndarray, sample_rate: int):
    return audio_data


def add_noise_with_snr_range(
    audio_data: np.ndarray, sample_rate: int, snr_range=(0, 10)
):
    # Calculate the power of the audio signal
    signal_power = np.mean(audio_data**2)

    # Choose a random SNR within the specified range
    chosen_snr = random.uniform(snr_range[0], snr_range[1])

    # Convert chosen SNR from dB to a linear scale
    chosen_snr_linear = 10 ** (chosen_snr / 10)

    # Calculate the required noise power to achieve the chosen SNR
    noise_power = signal_power / chosen_snr_linear

    # Generate white noise with the same shape as the audio data
    noise = np.random.normal(0, np.sqrt(noise_power), audio_data.shape)

    # Add the noise to the original audio
    noisy_audio_data = audio_data + noise / 2

    return noisy_audio_data


def crop_file(
    audio_data: np.ndarray,
    sample_rate: int,
    start_end_file: tuple = (0, 5.0),
):
    random_crop_idx = random.randint(0, 1)
    if random_crop_idx == 0:
        start_end_file_rand = (start_end_file[0] + 0.5, start_end_file[1])
    else:
        start_end_file_rand = (start_end_file[0], start_end_file[0] - 0.5)
    # Extract desired segments
    if start_end_file_rand != ():
        audio_data_cropped = audio_data[
            int(sample_rate * start_end_file_rand[0]) : int(
                sample_rate * start_end_file_rand[1]
            )
        ]
    return audio_data_cropped


def generate_caption(overlap: bool, file1_description: str, file2_description: str):
    """
    This function assumes that file1 plays first and file2 plays second.
    """
    dict_conj_and_prep = {
        "future": ["followed by", "before", "then"],
        "past": ["preceded by", "after"],
    }
    caption_coordinator_order = random.choice(["future", "past"])
    caption_conj_and_prep = random.choice(dict_conj_and_prep[caption_coordinator_order])
    if caption_coordinator_order == "future":
        caption = f"{file1_description.capitalize()} {caption_conj_and_prep} {file2_description}"
        return caption, caption_coordinator_order, caption_conj_and_prep, "12"
    else:
        caption = f"{file2_description.capitalize()} {caption_conj_and_prep} {file1_description}"
        return caption, caption_coordinator_order, caption_conj_and_prep, "21"


def librosa_combine_audio_files_adjusted(
    file1_path: Path,
    file2_path: Path,
    order: int,
    file1_description: str = "",
    file2_description: str = "",
    file1_transform=return_identical,
    file2_transform=return_identical,
    overlap: bool = False,
    overlap_time: float = 0.0,
    loudness_adjustment_dB: float = 0.0,
    loudness_relative_adjustment_dB: float = 0.0,
):
    """
    Combine two audio files into a 10-second long audio considering original loudness.

    Parameters:
    - file1_path: Path to the first audio file.
    - file2_path: Path to the second audio file.
    - order: Order of the audios. If 1, file1 plays first. If 2, file2 plays first.
    - file1/2_description: Description of each audio file.
    - overlap: Whether the audios should overlap.
    - overlap_time: Time in seconds for the overlap. If overlap > 10, assume both files start at second 0
    - loudness_relative_adjustment_dB: Additional loudness adjustment in dB for the second audio.
    """

    # Load the audio files
    y1, sr1 = librosa.load(file1_path, sr=None)
    y2, sr2 = librosa.load(file2_path, sr=None)

    # Ensure both files have the same sample rate
    if sr1 != sr2:
        raise ValueError("Sample rates of the two files do not match.")

    # Apply transformations
    y1 = file1_transform(y1, sr1)
    y2 = file2_transform(y2, sr2)

    # # Adjust loudness
    # y1 = librosa.effects.preemphasis(y1, coef=loudness_adjustment_dB)
    # y2 = librosa.effects.preemphasis(y2, coef=loudness_adjustment_dB + loudness_relative_adjustment_dB)

    # Combine based on order and overlap
    if order == 1:
        combined = combine_audio(y1, y2, sr1, overlap, overlap_time)
        caption, caption_coordinator_order, caption_conj_and_prep, caption_order = (
            generate_caption(overlap, file1_description, file2_description)
        )
    else:
        combined = combine_audio(y2, y1, sr1, overlap, overlap_time)
        caption, caption_coordinator_order, caption_conj_and_prep, caption_order = (
            generate_caption(overlap, file2_description, file1_description)
        )
    # Trim or extend to 10 seconds
    target_length = 10 * sr1
    combined = combined[:target_length]  # Trim to 10 seconds if longer
    if len(combined) < target_length:
        combined = np.pad(
            combined, (0, max(0, target_length - len(combined))), mode="constant"
        )

    return (
        combined,
        sr1,
        caption,
        caption_coordinator_order,
        caption_conj_and_prep,
        caption_order,
    )


def combine_audio(y1, y2, sr, overlap, overlap_time):
    if overlap:
        total_duration = min(sr * 10, len(y1) + len(y2) - int(sr * overlap_time))
        combined = np.zeros(total_duration)
        combined[: len(y1)] += y1
        combined[-len(y2) :] += y2
    else:
        if len(y1) + len(y2) < 10 * sr:
            silence = np.zeros(10 * sr - len(y1) - len(y2))
        else:
            silence = np.zeros(0)
        combined = np.concatenate([y1, silence, y2])
    return combined


def generate_joint_audio_and_captions_from_json(
    path_of_original_audios,
    path_to_save_audios,
    filename1,
    filename2,
    filename_to_category,
    category_to_description,
    random_order,
    overlap,
    overlap_time,
    file1_transform=return_identical,
    file2_transform=return_identical,
):
    (
        combined_audio,
        sr,
        _,
        caption_coordinator_order,
        caption_conj_and_prep,
        caption_order,
    ) = librosa_combine_audio_files_adjusted(
        file1_path=path_of_original_audios / filename1,
        file2_path=path_of_original_audios / filename2,
        file1_description=category_to_description[filename_to_category[filename1]],
        file2_description=category_to_description[filename_to_category[filename2]],
        order=random_order,
        overlap=overlap,
        overlap_time=overlap_time,
        file1_transform=file1_transform,
        file2_transform=file2_transform,
    )
    new_file_name = (
        filename1.split(".")[0]
        + "_"
        + filename2.split(".")[0]
        + "_"
        + caption_coordinator_order
        + "_"
        + caption_conj_and_prep.replace(" ", "_")
        + "_"
        + str(random_order)
        + "_"
        + caption_order
        + "_"
        + file1_transform.__name__
        + "_"
        + file2_transform.__name__
        + "_"
        + str(overlap)
        + ".wav"
    )
    if os.path.exists(path_to_save_audios / "joint_audios" / new_file_name) is False:
        sf.write(
            path_to_save_audios / "joint_audios" / new_file_name,
            combined_audio,
            sr,
        )


def main():
    path_of_original_audios = Path(
        "/scratch/shared/beegfs/yuki/data/esc50/ESC-50-master/audio"
    )
    path_to_save_audios = Path("/scratch/shared/nfs2/oncescu/shared-datasets/ESC-50")

    esc_50 = Path("/scratch/shared/beegfs/oncescu/shared-datasets/ESC-50")
    esc_50_csv = pd.read_csv(esc_50 / "meta/esc50.csv")

    # Loop through the ESC-50 CSV file and create a dictionary with the filename as the key and the category as the value
    filename_to_category = {}

    category_to_filename = {}
    for _, row_entry in esc_50_csv.iterrows():
        filename = row_entry["filename"]
        category = row_entry["category"]
        filename_to_category[filename] = category.replace("_", " ")
        if category.replace("_", " ") not in category_to_filename:
            category_to_filename[category.replace("_", " ")] = [filename]
        else:
            category_to_filename[category.replace("_", " ")].append(filename)

    # Load the json file containing the class name and the GPT generated potential description
    with open("../SynCaps/class_to_desc_gpt4.json", "r") as f:
        category_to_description = json.load(f)

    with open(path_to_save_audios / "audio_to_extra_info.json", "r") as f:
        filename_to_arguments = json.load(f)

    for _, arguments in tqdm.tqdm(filename_to_arguments.items()):
        generate_joint_audio_and_captions_from_json(
            path_of_original_audios=path_of_original_audios,
            path_to_save_audios=path_to_save_audios,
            filename1=arguments["file1"],
            filename2=arguments["file2"],
            random_order=arguments["order"],
            overlap=arguments["overlap"],
            overlap_time=arguments["overlap_time"],
            file1_transform=arguments["file1_transform"],
            file2_transform=arguments["file2_transform"],
            filename_to_category=filename_to_category,
            category_to_description=category_to_description,
        )


if __name__ == "__main__":
    main()
