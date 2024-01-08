import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math


# ノートナンバーから周波数へ
def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))


# ファイル名とパス
audio_file_path = "kimigayo.wav"

# 音声データの読み込み
y, sr = librosa.load(audio_file_path)

# 波形の表示
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# スペクトログラムの表示
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# SHSによる推定
fft_spec = np.fft.fft(y)
# "周波数らしさ(尤度)"の値を格納するリスト
likelihood_l = []


for nn in range(36,61):
    hz = nn2hz(nn)
    idx = hz / 8000 * len(fft_spec)
    power_s = 0
    for i, x in enumerate(fft_spec):
            power_s += x
            likelihood_l
