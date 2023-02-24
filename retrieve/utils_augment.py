import torch
from torch_audiomentations import Compose, Gain, PolarityInversion
from torch_audiomentations  import AddColoredNoise, Mix, AddBackgroundNoise
from torch_audiomentations import PitchShift, ApplyImpulseResponse, HighPassFilter
from torch_audiomentations import LowPassFilter
import json

def load_params(params_path):
    with open(params_path) as json_file:
        params_file= json.load(json_file)
    return params_file

 def define_augmentation(params):
    augmentation = Compose(
        transforms=[
            Gain( min_gain_in_db=params["min_gain_in_db"],
               max_gain_in_db=params["max_gain_in_db"],
               p=params["p_gain"],
               ),
            PolarityInversion(p=params["p_polarityinversion"]),
            AddColoredNoise(p=params["p_colourednoise"], min_snr_in_db=params["cn_min_snr_in_db"], max_snr_in_db = params["cn_max_snr_in_db"]),
            PitchShift(p=params["p_pitchshift"], min_transpose_semitones=params["min_transpose_semitones"],
                max_transpose_semitones= params["max_transpose_semitones"], sample_rate=48000),
            HighPassFilter(p=params["p_highpass"],min_cutoff_freq=params["high_min_cutoff_freq"], max_cutoff_freq=params["high_max_cutoff_freq"], sample_rate=48000),
            ApplyImpulseResponse(p=params["p_reverb"], ir_paths = "/gpfsscratch/rech/nou/uzn19yk/voxceleb/RIRS_NOISES/real_rirs_isotropic_noises",
                sample_rate=48000),
            LowPassFilter(p=params["p_low"], min_cutoff_freq=params["low_min_cutoff_freq"], max_cutoff_freq=params["low_max_cutoff_freq"])
            ]

    )
    return augmentation




def apply_augmentation(audio_samples, augmentation):
    return augmentation(audio_samples, sample_rate=48000)


def provide_comparison (learned_params, all_params):
    return None


