import numpy as np
import librosa
import matplotlib.pyplot as plt

# サンプリングレート
SR = 16000
# 窓をスライドさせる幅(フレームサイズ)
size_frame = 512
hamming_window = np.hamming(size_frame)
# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# 音声ファイルの読み込み
wav_a = "../wav/2_1_a.wav"
wav_i = "../wav/2_1_i.wav"
wav_u = "../wav/2_1_u.wav"
wav_e = "../wav/2_1_e.wav"
wav_o = "../wav/2_1_o.wav"

x_a, sr_a = librosa.load(wav_a, sr=SR)
x_i, sr_i = librosa.load(wav_i, sr=SR)
x_u, sr_u = librosa.load(wav_u, sr=SR)
x_e, sr_e = librosa.load(wav_e, sr=SR)
x_o, sr_o = librosa.load(wav_o, sr=SR)


def get_cepstrum(x):
    ceps_list = []
    for i in np.arange(0, len(x)-size_frame, size_shift):
        idx = int(i)  # arangeのインデクスはfloatなのでintに変換
        x_frame = x[idx: idx+size_frame]
        # print(x_frame)
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        # print(fft_spec)
        # 対数振幅スペクトルを計算
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        # ケプストラム分析
        ceps = np.fft.rfft(fft_log_abs_spec)
        ceps = ceps[:13]
        ceps = np.real(ceps)
        ceps_list.append(ceps)
    return ceps_list


# 対数尤度の計算
def calc_likelihood(x, mu, sigma):
    return -0.5 * np.sum((x - mu)**2 / sigma + np.log(sigma))
    # - 0.5 * np.log(2 * np.pi) * len(x)

cep_list_a = get_cepstrum(x_a)
cep_list_i = get_cepstrum(x_i)
cep_list_u = get_cepstrum(x_u)
cep_list_e = get_cepstrum(x_e)
cep_list_o = get_cepstrum(x_o)

mu_a = np.mean(np.array(cep_list_a), axis=0)
mu_i = np.mean(np.array(cep_list_i), axis=0)
mu_u = np.mean(np.array(cep_list_u), axis=0)
mu_e = np.mean(np.array(cep_list_e), axis=0)
mu_o = np.mean(np.array(cep_list_o), axis=0)

# print(mu_a, mu_i, mu_u, mu_e, mu_o)

sigma_a = np.var(np.array(cep_list_a), axis=0)
sigma_i = np.var(np.array(cep_list_i), axis=0)
sigma_u = np.var(np.array(cep_list_u), axis=0)
sigma_e = np.var(np.array(cep_list_e), axis=0)
sigma_o = np.var(np.array(cep_list_o), axis=0)

# print(sigma_a, sigma_i, sigma_u, sigma_e, sigma_o)
# 音声ファイルの読み込み
y, sr = librosa.load("output.wav", sr=SR)

pred = []
spectrogram = []

for i in np.arange(0, len(y)-size_frame, size_shift):
    idx = int(i)  # arangeのインデクスはfloatなのでintに変換
    y_frame = y[idx: idx+size_frame]
    fft_spec = np.fft.rfft(y_frame * hamming_window)
    # fft_log_abs_spec = np.log(np.abs(fft_spec))
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    # 計算した対数振幅スペクトログラムを配列に保存
    spectrogram.append(fft_log_abs_spec)
    # spectrogram.append(fft_log_abs_spec)
	
	# 【補足】
	# 配列（リスト）のデータ参照
	# list[A:B] listのA番目からB-1番目までのデータを取得

	# 窓掛けしたデータをFFT
	# np.fft.rfftを使用するとFFTの前半部分のみが得られる

	
	# np.fft.fft / np.fft.fft2 を用いた場合
	# 複素スペクトログラムの前半だけを取得
	#fft_spec_first = fft_spec[:int(size_frame/2)]

	# 【補足】
	# 配列（リスト）のデータ参照
	# list[:B] listの先頭からB-1番目までのデータを取得

	# 複素スペクトログラムを対数振幅スペクトログラムに


	# 低周波の部分のみを拡大したい場合
	# 例えば、500Hzまでを拡大する
	# また、最後のほうの画像描画処理において、
	# 	extent=[0, len(x), 0, 500], 
	# にする必要があることに注意
    cep = np.real(np.fft.rfft(fft_log_abs_spec))
    cep = cep[:13]
    likelihood_a = calc_likelihood(cep, mu_a, sigma_a)
    likelihood_i = calc_likelihood(cep, mu_i, sigma_i)
    likelihood_u = calc_likelihood(cep, mu_u, sigma_u)
    likelihood_e = calc_likelihood(cep, mu_e, sigma_e)
    likelihood_o = calc_likelihood(cep, mu_o, sigma_o)
    print(likelihood_a, likelihood_i, likelihood_u, likelihood_e, likelihood_o)
    likelihood = [likelihood_a, likelihood_i, likelihood_u, likelihood_e, likelihood_o]
    pred.append((likelihood.index(max(likelihood))+ 1)* 1000)
    
    # print(likelihood.index(max(likelihood)))


# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('time')		# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
    extent =[0, len(y), 0, SR/2], 			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.ylim(0, SR/2)

print(len(pred))
# Plot the predictions on top of the spectrogram
time_axis_pred = np.linspace(0, len(pred) * size_shift, num=len(pred))
plt.plot(time_axis_pred, pred, color="red")

# Show the plot
plt.show()
