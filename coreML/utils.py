import numpy as np
import librosa as lb


def audio2spec(filename: str, sr: int = 4000, n_mfcc: int = 30, n_fft: int = 400, 
                hop_length: int = 100) -> np.ndarray:
    soundArr, _ = lb.load(filename, sr=sr)
    S = np.abs(lb.stft(soundArr, n_fft=n_fft, hop_length=hop_length))
    powerSpec = lb.amplitude_to_db(S, ref=np.max)
    return powerSpec


def normalize(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr)
    std = np.std(arr)
    arr = (arr - mean) / std
    return arr


SPECTROGRAM_MAP = {'spec' : audio2spec}
