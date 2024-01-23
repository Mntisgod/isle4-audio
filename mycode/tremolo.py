import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile


def calc_loudness(x):
    # 音量(dB)を計算する
    # VoldB = 20 log10 RMS
    
    # サンプリングレート
    SR = 16000
    size_shift = 16000 / 100	# シフトサイズ = 0.01 秒 (10 msec)
    size_frame = 4096			# フレームサイズ
    is_spoken = False           # 発話しているかどうか
    volume = []			# 音量を保存するlist

    for i in np.arange(0, len(x)-size_frame, size_shift):
        # 該当フレームのデータを取得

        # arangeのインデクスはfloatなのでintに変換
        idx = int(i)
        x_frame = x[idx: idx+size_frame]
        # 音量
        vol = 20 * np.log10(np.mean(x_frame ** 2))
        if vol > -90 and not is_spoken:
            is_spoken = True
            print(f"start speaking:{i / SR} s")
        elif vol < -90 and is_spoken:
            is_spoken = False
            print(f"finish speaking:{ i/ SR} s")
            
        volume.append(vol)
    fig = plt.figure()

    # 縦軸を音量に
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('sec')
    ax1.set_ylabel('volume [dB]')
    ax1.plot(volume)
    # 表示
    plt.show()


SR = 16000
# 音声波形データを受け取る
wav_path = "hotaru_no_hikari.mp3"

# 音声ファイルの読み込み
x, _ = librosa.load(wav_path, sr=SR)

# トレモロの係数
D = 0.5
R = 0.5

# トレモロの係数をかける
x_changed = x * (1 + D * np.sin(2 * np.pi * R * np.arange(len(x)) / SR))

# 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
x_changed = (x_changed * 32768.0). astype('int16')

# 音声ファイルとして出力する
filename = 'voice_change.wav'
scipy.io.wavfile.write(filename , int(SR), x_changed)

# 画像として音量を出力
calc_loudness(x)
calc_loudness(x_changed)
