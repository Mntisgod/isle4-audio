# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa


def main():
    # 音量(dB)を計算する
    # VoldB = 20 log10 RMS
    
    # サンプリングレート
    SR = 16000
    size_shift = 16000 / 100	# シフトサイズ = 0.01 秒 (10 msec)
    size_frame = 4096			# フレームサイズ
    is_spoken = False           # 発話しているかどうか
    volume = []			# 音量を保存するlist

    # 音声ファイルの読み込み
    wav_name = str(input())
    x, _ = librosa.load(wav_name, sr=SR)

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
    # plt.show()
    # 画像ファイルに保存
    fig.savefig(wav_name.replace('.wav', '') + '-plot-loudness.png')


if __name__ == '__main__':
    main()
