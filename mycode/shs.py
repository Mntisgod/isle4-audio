import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math


SR = 16000
size_frame = 2048
hamming_window = np.hamming(size_frame)
size_shift = int(16000 / 3)  # Ensure shift size is an integer

note_dict = {
    35: 'None',
    36: 'C2',
    37: 'C#2',
    38: 'D2',
    39: 'D#2',
    40: 'E2',
    41: 'F2',
    42: 'F#2',
    43: 'G2',
    44: 'G#2',
    45: 'A2',
    46: 'A#2',
    47: 'B2',
    48: 'C3',
    49: 'C#3',
    50: 'D3',
    51: 'D#3',
    52: 'E3',
    53: 'F3',
    54: 'F#3',
    55: 'G3',
    56: 'G#3',
    57: 'A3',
    58: 'A#3',
    59: 'B3',
    60: 'C4',
}


# ノートナンバーから周波数へ
def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))


def shs(fft_spec):
    likelihood_l = []
    frequencies = np.linspace(8000/len(fft_spec), 8000, len(fft_spec))
    
    for nn in range(36, 61):
        print(nn)
        hz = nn2hz(nn)
        print(hz)
        power_s = 0
        for i in range(1, int(8000/hz)):
            f = hz * i
            freq_diff = np.abs(frequencies - f)
            # Use np.where to find indices where the condition is satisfied
            indices = np.where(freq_diff < 0.03)[0]
            # print(indices)
            if len(indices) == 0:
                continue
            power_s += fft_spec[indices[0]] ** 2
        # print(power_s)
        likelihood_l.append(power_s)
    
    if max(likelihood_l) <= 0.000:
        return 35
    return likelihood_l.index(max(likelihood_l)) + 36


# ファイル名とパス
audio_file_path = "shs-test.wav"

# 音声データの読み込み
y, sr = librosa.load(audio_file_path, sr=SR)

# 波形の表示
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.show()

# スペクトログラムの表示
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=SR, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
# plt.show()

fundamental_freq_list = []
# SHSによる推定
for i in range(0, len(y)-size_frame, size_shift):
    idy = int(i)
    y_frame = y[idy:idy+size_frame]
    fft_spec = np.fft.rfft(y_frame * hamming_window)
    fundamental_freq = shs(fft_spec)
    # dictからノートナンバーを取得
    note = note_dict[fundamental_freq]
    fundamental_freq_list.append(fundamental_freq)

# ピッチの表示
plt.figure(figsize=(10, 4))
plt.plot(fundamental_freq_list)
plt.title('Pitch')
plt.xlabel('Time (s)')
plt.ylabel('Pitch')
plt.show()
