import numpy as np
import math
import matplotlib.pyplot as plt
import librosa


SR = 16000
KER_SIZE = 1000


def nmf(spec):
    # Initialize H and U
    H = np.abs(np.random.rand(len(spec), KER_SIZE))
    U = np.abs(np.random.rand(len(spec[0]), KER_SIZE))

    # Update H and U
    print(len(spec), len(spec[0]))
    for i in range(100):
        # U = U * np.dot(H.T, spec) / np.dot(np.dot(H.T, H), U)
        # H = H * np.dot(spec, U.T) / np.dot(np.dot(H, U), U.T)
        H = H * np.dot(spec, U) / np.dot(np.dot(H, U.T), U)
        U = U * np.dot(H.T, spec).T / np.dot(H.T, np.dot(H, U.T)).T
    return H, U


def pltNMF(H, U):
    plt.subplot(2, 1, 1)
    plt.plot(H)
    plt.subplot(2, 1, 2)
    plt.plot(U)
    plt.show()


wav_path = "easy_chords.wav"
x, sr = librosa.load(wav_path, sr=SR)

size_frame = 512
hamming_window = np.hamming(size_frame)
size_shift = int(16000 / 100)  # Ensure shift size is an integer

spectrogram = []

for i in range(0, len(x)-size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx:idx+size_frame]
    fft_spec = np.abs(np.fft.rfft(x_frame * hamming_window))
    spectrogram.append(fft_spec)

print(spectrogram)
h, u = nmf(spectrogram)
pltNMF(h, u)
