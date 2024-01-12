import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import sys


SR = 44100
KER_SIZE = 10


def nmf(spec):
    # Initialize H and U
    H = np.abs(np.random.rand(len(spec), KER_SIZE))
    U = np.abs(np.random.rand(len(spec[0]), KER_SIZE))

    # Update H and U
    for i in range(100):
        # U = U * np.dot(H.T, spec) / np.dot(np.dot(H.T, H), U)
        # H = H * np.dot(spec, U.T) / np.dot(np.dot(H, U), U.T)
        H = H * np.dot(spec, U) / np.dot(np.dot(H, U.T), U)
        U = U * np.dot(H.T, spec).T / np.dot(H.T, np.dot(H, U.T)).T
    return H.T, U


def pltNMF(H, U):
    plt.subplot(2, 1, 1)
    plt.plot(H)
    plt.subplot(2, 1, 2)
    plt.plot(U)
    plt.show()


wav_path = "_kongyo_1.wav"
x, _ = librosa.load(wav_path, sr=SR)

size_frame = 512
hamming_window = np.hamming(size_frame)
size_shift = 44100 / 100  # Ensure shift size is an integer

spectrogram = []

for i in np.arange(0, len(x)-size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx:idx+size_frame]
    fft_spec = np.abs(np.fft.fft(x_frame))
    fft_spec = [np.log(i) if i > 0 else 0.00000000001 for i in fft_spec]
    spectrogram.append(fft_spec)

#
# H と U を可視化
#
U, H = nmf(spectrogram)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('H')
plt.xlabel('Component')
plt.ylabel('Frequency [Hz]')
plt.imshow(
    H,
    aspect='auto',
    origin='lower',
    interpolation='none',
    extent=[0, H.shape[1], 0, SR//2]
)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('U')
plt.xlabel('Time [frame]')
plt.ylabel('Component')
plt.imshow(
    U,
    aspect='auto',
    origin='lower',
    interpolation='none', 
    extent=[0, U.shape[1], 0, U.shape[0]]
)
plt.colorbar()
plt.show()
