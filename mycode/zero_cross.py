#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ゼロ交差数を計算する関数
#

import numpy as np
import librosa
import matplotlib.pyplot as plt


# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
	
	zc = 0

	for i in range(len(waveform) - 1):
		if (
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	
	return zc


# 音声波形データを受け取り，ゼロ交差数を計算する関数
# 簡潔版
def zero_cross_short(waveform):
	
	d = np.array(waveform)
	return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])


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

wav_path = "easy_chords.wav"
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
size_shift = 16000 / 100  # 例として160サンプルごとに区間を分割
size_frame = 512
# スペクトログラムを保存するlist
spectrogram = []
fundamental_frequency = []
zc_list = []
hamming_window = np.hamming(size_frame)

# 各区間での周波数を計算
for i, j in enumerate(np.arange(0, len(x)-size_frame, size_shift)):
	# 該当フレームのデータを取得
	idx = int(j)  # arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx: idx+size_frame]
	fft_spec = np.fft.rfft(x_frame * hamming_window)
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	# size_target = int(len(fft_log_abs_spec) * (500 / (SR/2)))
	# fft_log_abs_spec = fft_log_abs_spec[:size_target]
	spectrogram.append(fft_log_abs_spec)

	start_idx = (i) * size_shift
	end_idx = int(i + 1) * size_shift
	autocorr_interval = autocorr[int(start_idx):int(end_idx)]

	# 自己相関が最大となるインデックスを得る
	max_peak_index_interval = np.argmax(autocorr_interval)

	# 区間ごとの周波数を計算して出力
	freq_interval = SR / max_peak_index_interval
	fundamental_frequency.append(freq_interval)
	# print(f"自己相関を用いた場合: 区間 {i + 1}: {freq_interval} Hz")

	# 音声波形データを受け取り，ゼロ交差数を計算する
	x_interval = x[int(start_idx):int(end_idx)]
	# print(f"ゼロ交差数を用いた場合: 区間 {i + 1}: {zc/2} Hz")
	# if()
	zc = zero_cross_short(x_interval)
	zc_list.append(zc/2)

	# プロットするための配列を作成
	# 1サンプルごとに時間を増やしていく
	# time = np.arange(len(x)) / SR
	# # プロット
	# plt.plot(time, x)


#　ゼロ交差数の正規化
zc_list = np.array(zc_list)
zc_list = zc_list / np.max(zc_list)

fundamental_frequency = np.where(zc_list > 0.15, 0, fundamental_frequency)
# スペクトログラムのプロット
fig = plt.figure()
plt.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, len(x)/SR, 0, SR/2], 			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.title('Spectrogram')
plt.xlabel('Time (s)')
1
# ゼロ交差数を用いた基本周波数のプロット
# plt.imshow(
# 	np.arange(0, len(x), size_shift)[:len(fundamental_frequency)] / SR,
# 	extent=[0, len(x)/SR, 0, 500], 			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
# 	aspect='auto',
# 	# color='red',
# 	interpolation='nearest'
# )
fundamental_frequency = np.where(fundamental_frequency > 8000, 8000, fundamental_frequency)
plt.plot(np.arange(0, len(x), size_shift)[:len(fundamental_frequency)] / SR, fundamental_frequency, color="red", label='Fundamental Frequency')
plt.ylabel('Frequency (Hz)')

# 画像ファイルに保存
fig.savefig(wav_path.replace('.wav', '') + '-plot-spectrum-and-fundamental.png')

# 表示
plt.show()

# # 画像として保存するための設定
# fig = plt.figure()

# # スペクトログラムを描画
# plt.xlabel('sample')					# x軸のラベルを設定
# plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
# plt.imshow(
# 	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
#     extent =[0, len(x), 0, SR/2], 			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
# 	aspect='auto',
# 	interpolation='nearest'
# )
# plt.plot(x, fundamental_frequency, color="red", label='基本周波数')
# # 画像ファイルに保存
# fig.savefig(wav_path.replace('.wav', '') + '-plot-spectrum-whole.png')


# wav_name = "../wav/aiueo.wav"
# # x, _ = librosa.load(wav_name, sr=SR)
# zero_cross_short(x)

