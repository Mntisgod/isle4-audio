# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math


# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
    # fが無限大の場合はエラーとなるので，その場合は-1を返す
    if frequency == float('inf'):
        return -1
    return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
    # （自分で実装すること，passは消す）
    if index == 0 or index == len(a)-1:
        return False
    if a[index-1] < a[index] and a[index] > a[index+1]:
        return True
    else:
        return False

# サンプリングレート
SR = 16000

wav_path = "shs-test.wav"
# 音声ファイルの読み込み
x, _ = librosa.load(wav_path, sr=SR)

# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
autocorr = np.correlate(x, x, 'full')

# 不要な前半を捨てる
autocorr = autocorr[len(autocorr) // 2:]

# ピークのインデックスを抽出する
peakindices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]
# インデックス0 がピークに含まれていれば捨てる
peakindices = [i for i in peakindices if i != 0]

# 一定間隔で分割する
interval_size = 1000  # 例として1000サンプルごとに区間を分割
num_intervals = len(x) // interval_size

nn_list = []


# 各区間での周波数を計算
for i in range(num_intervals):
    start_idx = i * interval_size
    end_idx = (i + 1) * interval_size

    # 区間ごとの自己相関を取得
    autocorr_interval = autocorr[start_idx:end_idx]

    # 自己相関が最大となるインデックスを得る
    max_peak_index_interval = np.argmax(autocorr_interval)

    # 区間ごとの周波数を計算して出力
    freq_interval = SR / max_peak_index_interval
    nn_list.append(hz2nn(freq_interval))

# ピッチの表示
plt.figure(figsize=(10, 4))
plt.plot(nn_list)
plt.title('Pitch')
plt.xlabel('Time (s)')
plt.ylabel('Pitch')
plt.show()