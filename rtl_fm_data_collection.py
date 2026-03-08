from rtlsdr import RtlSdr, RtlSdrAio
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import firwin, resample_poly, resample
from pathlib import Path
import h5py

import random

import pdb


def process_signal(
    signal: np.ndarray,
    sample_rate_hz: float,
    filter_bw_hz: float = 200e3,
    decimation_factor: int = 5,
) -> None:
    frac_filter_freq = filter_bw_hz / sample_rate_hz

    h = firwin(512, frac_filter_freq)
    filter_signal = np.convolve(signal, h, mode="same")
    
    return filter_signal[::decimation_factor]


def collect_data(
    center_freq_hz: float = 90.9e6, # 96.5 101.1 102.1
    sample_rate_hz: float = 1.024e6,
    gain: float = 50.0,
    snippet_duration_sec: float = 30.0,
    save_root: Path = Path("/home/brandon/git/notebooks/data/fm-iq/v1/"),
    num_data_samples: int = 1
) -> None:
    radio = RtlSdr()
    
    radio.center_freq = center_freq_hz
    radio.sample_rate = sample_rate_hz
    radio.gain = gain
    
    radio.read_samples(2048) # Clear buffer.

    for idx in range(1, num_data_samples+1):
        if idx % 100 == 0:
            print(f"[{idx:6}/{num_data_samples}]")

        # Loop and receive data.
        recv_num_samples: int = int(snippet_duration_sec * radio.sample_rate)
        signal = radio.read_samples(recv_num_samples)
        signal = process_signal(signal, radio.sample_rate)
        
        file_path = save_root.joinpath(f"samp-{idx}.npy")
        np.save(file_path, signal)
    radio.close()


if __name__ == "__main__":
    collect_data()
