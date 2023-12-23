#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
wav_name = "../wav/aiueo.wav"
x, _ = librosa.load(wav_name, sr=SR)


# スペクトルを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()
plt.plot(x, color="red")
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

plt.show()

# 画像ファイルに保存
fig.savefig(wav_name.replace('.wav', '') + '-plot-wav.png')