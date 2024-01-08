import numpy as np
import math
import matplotlib.pyplot as plt
import librosa


def hz2nn(frequency):
    if frequency <= 0:
        return 0  # or any default value you prefer when frequency is non-positive
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69


def generate_chroma_vector(spectrum, frequencies):
    cv = np.zeros(12)
    for s, f in zip(spectrum, frequencies):
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)
    return cv / np.linalg.norm(cv)  # Normalize the chroma vector


SR = 16000
wav_path = "easy_chords.wav"
x, sr = librosa.load(wav_path, sr=SR)

size_frame = 2048
hamming_window = np.hamming(size_frame)
size_shift = int(16000 / 100)  # Ensure shift size is an integer

spectrogram = []
chromagram = []
max_chroma_list = []

for i in range(0, len(x)-size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx:idx+size_frame]
    frequencies = np.fft.fftfreq(size_frame, d=1/SR)
    _chroma_vector = generate_chroma_vector(np.fft.rfft(x_frame * hamming_window), frequencies)
    # 和音推定
    ans = 0
    for i in range(12):
        # Major
        if ans < _chroma_vector[i] + 0.5 * _chroma_vector[(i+4) % 12] + 0.8 *_chroma_vector[(i+7) % 12]:
            max_chroma = i
            ans = _chroma_vector[i] + 0.5 * _chroma_vector[(i+4) % 12] + 0.8 *_chroma_vector[(i+7) % 12]
    for i in range(12):
        if ans < _chroma_vector[i] + 0.5 * _chroma_vector[(i+3) % 12] + 0.8 * _chroma_vector[(i+7) % 12]:
            max_chroma = 12 + i
            ans = _chroma_vector[i] + 0.5 * _chroma_vector[(i+3) % 12] + 0.8 * _chroma_vector[(i+7) % 12]

    max_chroma_list.append(max_chroma)
    chromagram.append(_chroma_vector)
    fft_log_abs_spec = np.log(np.abs(np.fft.rfft(x_frame * hamming_window)))
    spectrogram.append(fft_log_abs_spec)

fig = plt.figure()
plt.xlabel('sample')
plt.ylabel('frequency [Hz]')
plt.imshow(
    np.flipud(np.array(spectrogram).T),
    extent=[0, len(x), 0, SR/2],
    aspect='auto',
    interpolation='nearest'
)
plt.show()

fig = plt.figure()
plt.xlabel('time (s)')
plt.ylabel('chroma')
plt.imshow(
    np.flipud(np.array(chromagram).T),
    extent=[0, len(x)/SR, 0, 12],
    aspect='auto',
    interpolation='nearest'
)
plt.show()

#max chromaを表示
fig = plt.figure()
plt.xlabel('time (s)')
plt.ylabel('max chroma')
plt.plot(max_chroma_list)
plt.show()

fig.savefig('plot-spectrogram.png')
