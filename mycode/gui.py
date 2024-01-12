#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込みスペクトログラムを表示する
# その隣に時間を選択するスライドバーと選択した時間に対応したスペクトルを表示する
# GUIのツールとしてTkinterを使用する
#

# ライブラリの読み込み
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

size_frame = 4096    # フレームサイズ
SR = 16000            # サンプリングレート
size_shift = SR / 100    # シフトサイズ = 0.01 秒 (10 msec)


def open_file(event):
    # tkinterで読み込む
    fTyp = [("wav file", "*.wav")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    global input_file
    input_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    # 音声ファイルを読み込む
    x, _ = librosa.load(input_file, sr=SR)

    # ファイルサイズ（秒）
    duration = len(x) / SR

    # ハミング窓
    hamming_window = np.hamming(size_frame)   

    def calc(x):
        def is_peak(a, index):
            if index == 0 or index == len(a)-1:
                return False
            if a[index-1] < a[index] and a[index] > a[index+1]:
                return True
            else:
                return False
        
        # 対数尤度の計算
        def calc_likelihood(x, mu, sigma):
            return -0.5 * np.sum((x - mu)**2 / sigma + np.log(sigma))

        def vowel_recognition(x):
            # ケプストラム計算
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

            # - 0.5 * np.log(2 * np.pi) * len(x)
            # 音声ファイルの読み込み
            wav_a = "../source/2_1_a.wav"
            wav_i = "../source/2_1_i.wav"
            wav_u = "../source/2_1_u.wav"
            wav_e = "../source/2_1_e.wav"
            wav_o = "../source/2_1_o.wav"

            x_a, sr_a = librosa.load(wav_a, sr=SR)
            x_i, sr_i = librosa.load(wav_i, sr=SR)
            x_u, sr_u = librosa.load(wav_u, sr=SR)
            x_e, sr_e = librosa.load(wav_e, sr=SR)
            x_o, sr_o = librosa.load(wav_o, sr=SR)

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

            var_a = np.var(np.array(cep_list_a), axis=0)
            var_i = np.var(np.array(cep_list_i), axis=0)
            var_u = np.var(np.array(cep_list_u), axis=0)
            var_e = np.var(np.array(cep_list_e), axis=0)
            var_o = np.var(np.array(cep_list_o), axis=0)

            return mu_a, mu_i, mu_u, mu_e, mu_o, var_a, var_i, var_u, var_e, var_o

        mu_a, mu_i, mu_u, mu_e, mu_o, var_a, var_i, var_u, var_e, var_o = vowel_recognition(x)
        # スペクトログラムを保存するlist
        spectrogram = []
        hz_list = []
        pred = []
        autocorr = np.correlate(x, x, 'full')

        # 不要な前半を捨てる
        autocorr = autocorr[len(autocorr) // 2:] 
        # フレーム毎にスペクトルを計算
        for i in np.arange(0, len(x)-size_frame, size_shift):
            
            # 該当フレームのデータを取得
            start_idx = int(i)    # arangeのインデクスはfloatなのでintに変換
            end_idx = start_idx+size_frame
            x_frame = x[start_idx: end_idx]

            # スペクトル
            fft_spec = np.fft.rfft(x_frame * hamming_window)
            fft_log_abs_spec = np.log(np.abs(fft_spec))
            spectrogram.append(fft_log_abs_spec)
            
            # 基本周波数
            # 区間ごとの自己相関を取得
            autocorr_interval = autocorr[start_idx:end_idx]
                # ピークのインデックスを抽出する
            peakindices = [i for i in range(len(autocorr_interval)) if is_peak(autocorr_interval, i)]
                # インデックス0 がピークに含まれていれば捨てる
            peakindices = [i for i in peakindices if i != 0]
                # 自己相関が最大となるインデックスを得る
            max_peak_index = max(peakindices, key=lambda index: autocorr_interval[index])
            # max_peak_index_interval = np.argmax(autocorr_interval)
            # 区間ごとの周波数を計算して出力
            freq_interval = SR / max_peak_index
            hz_list.append(freq_interval)

            # 母音の判定
            cep = np.real(np.fft.rfft(fft_log_abs_spec))
            cep = cep[:13]
            likelihood_a = calc_likelihood(cep, mu_a, var_a)
            likelihood_i = calc_likelihood(cep, mu_i, var_i)
            likelihood_u = calc_likelihood(cep, mu_u, var_u)
            likelihood_e = calc_likelihood(cep, mu_e, var_e)
            likelihood_o = calc_likelihood(cep, mu_o, var_o)
            likelihood = [likelihood_a, likelihood_i, likelihood_u, likelihood_e, likelihood_o]
            pred.append((likelihood.index(max(likelihood))+ 1)* SR / 10)

        return spectrogram, hz_list, pred

    spectrogram, hz_list, pred = calc(x)

    # Tkinterのウィジェットを階層的に管理するためにFrameを使用
    # frame1 ... スペクトログラムを表示
    # frame2 ... Scale（スライドバー）とスペクトルを表示
    frame1 = tkinter.Frame(root)
    frame2 = tkinter.Frame(root)
    frame1.pack(side="left")
    frame2.pack(side="left")

    def _draw_data(spectrogram, hz_list, pred):
        # まずはスペクトログラムを描画
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=frame1)    # masterに対象とするframeを指定
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('sec')
        ax1.set_ylabel('frequency [Hz]')
        ax1.imshow(
            np.flipud(np.array(spectrogram).T),
            extent=[0, duration, 0, 8000],
            aspect='auto',
            interpolation='nearest'
        )
        # 続いて右側のy軸を追加して，音量を重ねて描画
        ax3 = ax1.twinx()
        ax3.set_ylabel('frequency [Hz]')
        x_data = np.linspace(0, duration, len(hz_list))
        ax3.plot(x_data, hz_list, c='b')
        ax3.plot(x_data, pred, c='r')
        # time_axis_pred = np.linspace(0, len(pred) * size_shift, num=len(pred))
        # plt.plot(time_axis_pred, pred, color="red")
        canvas.get_tk_widget().pack(side="left")    # 最後にFrameに追加する処理

    _draw_data(spectrogram, hz_list, pred)


    # スライドバーの値が変更されたときに呼び出されるコールバック関数
    # ここで右側のグラフに
    # vはスライドバーの値
    def _draw_spectrum(v):

        # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
        index = int((len(spectrogram)-1) * (float(v) / duration))
        spectrum = spectrogram[index]

        # 直前のスペクトル描画を削除し，新たなスペクトルを描画
        plt.cla()
        x_data = np.fft.rfftfreq(size_frame, d=1/SR)
        ax2.plot(x_data, spectrum)
        ax2.set_ylim(-10, 5)
        ax2.set_xlim(0, SR/2)
        ax2.set_ylabel('amblitude')
        ax2.set_xlabel('frequency [Hz]')
        canvas2.draw()


    # スペクトルを表示する領域を確保
    # ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
    fig2, ax2 = plt.subplots()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.get_tk_widget().pack(side="top")    # "top"は上部方向にウィジェットを積むことを意味する

    # スライドバーを作成
    scale = tkinter.Scale(
        command=_draw_spectrum,        # ここにコールバック関数を指定
        master=frame2,                # 表示するフレーム
        from_=0,                    # 最小値
        to=duration,                # 最大値
        resolution=size_shift/SR,    # 刻み幅
        label=u'スペクトルを表示する時間[sec]',
        orient=tkinter.HORIZONTAL,    # 横方向にスライド
        length=600,                    # 横サイズ
        width=50,                    # 縦サイズ
        font=("", 20)                # フォントサイズは20pxに設定
    )
    scale.pack(side="top")

# def play_music(play = True):
#     if play :


#     else :

# Tkinterを初期化
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-SAMPLE")

# ファイルを開く
open_file_button = tkinter.Button(root, text='Open file')
open_file_button.pack(side=tkinter.TOP)
open_file_button.bind('<Button-1>', open_file)

# # 再生ボタン
# play_music_button = tkinter.Button(root, text='Play')
# play_music_button.pack(side=tkinter.END)
# play_music_button.bind('<Button-1>', play_music)


tkinter.mainloop()