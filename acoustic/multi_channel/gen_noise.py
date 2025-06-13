import argparse
import os
import subprocess
import re
import soundfile as sf

#import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
from numpy.random import uniform
import random
#from rir_simulate import rir_room
import math
import torchaudio

import torch
import random
import numpy as np
import scipy
from scipy.signal import stft, istft,get_window
from scipy.linalg import cholesky, eigh


def gen_desired_spatial_coherence(pos_mics: np.ndarray, fs: int, noise_field: str = 'spherical', c: float = 343.0, nfft: int = 256) -> np.ndarray:
    """generate desired spatial coherence for one array

    Args:
        pos_mics: microphone positions, shape (num_mics, 3)
        fs: sampling frequency
        noise_field: 'spherical' or 'cylindrical'
        c: sound velocity
        nfft: points of fft

    Raises:
        Exception: Unknown noise field if noise_field != 'spherical' and != 'cylindrical'

    Returns:
        np.ndarray: desired spatial coherence, shape [num_mics, num_mics, num_freqs]
        np.ndarray: desired mixing matrices, shape [num_freqs, num_mics, num_mics]


    Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
    """
    #assert pos_mics.shape[1] == 7, pos_mics.shape

    M = pos_mics.shape[0]
    num_freqs = nfft // 2 + 1

    # compute desired spatial coherence matric
    ww = 2 * math.pi * fs * np.array(list(range(num_freqs))) / nfft
    dist = np.linalg.norm(pos_mics[:, np.newaxis, :] - pos_mics[np.newaxis, :, :], axis=-1, keepdims=True)
    if noise_field == 'spherical':
        DSC = np.sinc(ww * dist / (c * math.pi))
    elif noise_field == 'cylindrical':
        DSC = scipy.special.j0(ww * dist / c)
    else:
        raise Exception('Unknown noise field')

    # compute mixing matrices of the desired spatial coherence matric

    Cs = np.zeros((num_freqs, M, M), dtype=np.complex128)
    for k in range(1, num_freqs):
        
        DSC[:, :, k] = (DSC[:, :, k] + DSC[:, :, k].conj().T) / 2
        
        D, V = eigh(DSC[:, :, k])
        D = np.clip(D, a_min=1e-8, a_max=None)
        C = V @ np.diag(np.sqrt(D))

        Cs[k, ...] = C

    return DSC, Cs


def gen_diffuse_noise(noise: np.ndarray, L: int, Cs: np.ndarray, nfft: int = 256, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """generate diffuse noise with the mixing matrice of desired spatial coherence

    Args:
        noise: at least `num_mic*L` samples long
        L: the length in samples
        Cs: mixing matrices, shape [num_freqs, num_mics, num_mics]
        nfft: the number of fft points
        rng: random number generator used for reproducibility

    Returns:
        np.ndarray: multi-channel diffuse noise, shape [num_mics, L]
    """

    M = Cs.shape[-1]

    assert noise.shape[-1] >= M * L, ("The noise signal should be at least `num_mic*L` samples long", noise.shape, M, L)
    start = rng.integers(low=0, high=noise.shape[-1] - M * L + 1)
    noise = noise[start:start + M * L].reshape(M, L)
    noise = noise - np.mean(noise, axis=-1, keepdims=True)


    f, t, N = stft(noise, fs = 16000,window='hann', nperseg=nfft, noverlap=int(0.75 * nfft), nfft=nfft)  # N: [M,F,T]
    # Generate output in the STFT domain for each frequency bin k
    X = np.einsum('fmn,mft->nft', np.conj(Cs), N)
    # Compute inverse STFT
    F, x = istft(X, window='hann', nperseg=nfft, noverlap=0.75 * nfft, nfft=nfft)
    x = x[:, :L]
    return x  # [M, L]

