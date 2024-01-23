import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math


SR = 16000
size_frame = 4096
hamming_window = np.hamming(size_frame)
size_shift = int(16000 / 100)  # Ensure shift size is an integer

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

def nn2key(notenum):
    if notenum <= 23:
        return '0'
    key = notenum % 12
    if key == 0:
        return 'C' + str(notenum // 12 - 1)
    elif key == 1:
        return 'C#' + str(notenum // 12 - 1)
    elif key == 2:
        return 'D' + str(notenum // 12 - 1)
    elif key == 3:
        return 'D#' + str(notenum // 12 - 1)
    elif key == 4:
        return 'E' + str(notenum // 12 - 1)
    elif key == 5:
        return 'F' + str(notenum // 12 - 1)
    elif key == 6:
        return 'F#' + str(notenum // 12 - 1)
    elif key == 7:
        return 'G' + str(notenum // 12 - 1)
    elif key == 8:
        return 'G#' + str(notenum // 12 - 1)
    elif key == 9:
        return 'A' + str(notenum // 12 - 1)
    elif key == 10:
        return 'A#' + str(notenum // 12 - 1)
    elif key == 11:
        return 'B' + str(notenum // 12 - 1)
    else:
        return 'error'


# ノートナンバーから周波数へ
def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))


def shs(fft_spec):
    likelihood_l = []
    frequencies = np.linspace(8000/len(fft_spec), 8000, len(fft_spec))
    
    for nn in range(36, 61):
        hz = nn2hz(nn)
        power_s = 0
        for i in range(1, int(8000/hz)):
            f = hz * i      
            freq_diff = np.abs(frequencies - f)
            target_index = np.argmin(freq_diff)
            if target_index >= len(fft_spec):
                continue
            # target_indexのビンのパワーを加算
            power_s += np.abs(fft_spec[target_index])
            if target_index > 0:
                power_s += np.abs(fft_spec[target_index-1])
            if target_index < len(fft_spec)-1:
                power_s += np.abs(fft_spec[target_index+1])
        # print(power_s)
        likelihood_l.append(power_s)3
    
    if max(likelihood_l) <= 0.000:
        return 35

    return likelihood_l.index(max(likelihood_l)) + 36


# ファイル名とパス
audio_file_path = "hotaru_no_hikari.mp3"

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
    fft_spec = np.fft.fft(y_frame)
    fundamental_freq = shs(fft_spec)
    # dictからノートナンバーを取得
    note = note_dict[fundamental_freq]
    fundamental_freq_list.append(fundamental_freq)
    print(f"SHS: {note} ")

# ピッチの表示
plt.figure(figsize=(10, 4))
plt.yticks(np.arange(36, 61, 1), [nn2key(nn) for nn in range(36, 61)])
plt.plot(fundamental_freq_list)
plt.title('Pitch')
plt.xlabel('Time (s)')
plt.ylabel('Pitch')
plt.show()
