#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード

# 音声ファイルを読み込みスペクトログラムを表示する
# その隣に時間を選択するスライドバーと選択した時間に対応したスペクトルを表示する
# GUIのツールとしてTkinterを使用する
#

# ライブラリの読み込み
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import threading
import time
import os
import pyaudio
from pydub.utils import make_chunks
from pydub import AudioSegment
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

size_frame = 4096    # フレームサイズ
SR = 16000        # サンプリングレート
size_shift = SR / 100    # シフトサイズ = 0.01 秒 (10 msec)
x = None           # 音声波形データ
is_playing = False    # 再生中かどうか
hamming_window = np.hamming(size_frame)
EPSILON = 1e-10
# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(size_frame / 2)
# 楽曲のデータを格納
x_stacked_data_music = np.array([])
# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100
# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False


def open_file(event):
    # tkinterで読み込む
    fTyp = [("wav file", "*.wav")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    global input_file
    print("open")
    input_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    # 音声ファイルを読み込む
    global x
    print("open2")
    x, _ = librosa.load(input_file, sr=SR)
    global audio_data
    print("open3")
    audio_data = AudioSegment.from_wav(input_file)
    print(type(audio_data))
    print("open4")
    global duration
    # ファイルサイズ（秒）
    duration = len(x) / SR

    # ハミング窓

    def calc(x):
        def is_peak(a, index):
            if index == 0 or index == len(a)-1:
                return False
            if a[index-1] < a[index] and a[index] > a[index+1]:
                return True
            else:
                return False
        
        # # 対数尤度の計算
        # def calc_likelihood(x, mu, sigma):
        #     return -0.5 * np.sum((x - mu)**2 / sigma + np.log(sigma))

        # def vowel_recognition(x):
        #     # ケプストラム計算
        #     def get_cepstrum(x):
        #         ceps_list = []
        #         for i in np.arange(0, len(x)-size_frame, size_shift):
        #             idx = int(i)  # arangeのインデクスはfloatなのでintに変換
        #             x_frame = x[idx: idx+size_frame]
        #             # print(x_frame)
        #             fft_spec = np.fft.rfft(x_frame * hamming_window)
        #             # print(fft_spec)
        #             # 対数振幅スペクトルを計算
        #             fft_log_abs_spec = np.log(np.abs(fft_spec))
        #             # ケプストラム分析
        #             ceps = np.fft.rfft(fft_log_abs_spec)
        #             ceps = ceps[:13]
        #             ceps = np.real(ceps)
        #             ceps_list.append(ceps)
        #         return ceps_list

        #     # - 0.5 * np.log(2 * np.pi) * len(x)
        #     # 音声ファイルの読み込み
        #     wav_a = "../source/2_1_a.wav"
        #     wav_i = "../source/2_1_i.wav"
        #     wav_u = "../source/2_1_u.wav"
        #     wav_e = "../source/2_1_e.wav"
        #     wav_o = "../source/2_1_o.wav"

        #     x_a, sr_a = librosa.load(wav_a, sr=SR)
        #     x_i, sr_i = librosa.load(wav_i, sr=SR)
        #     x_u, sr_u = librosa.load(wav_u, sr=SR)
        #     x_e, sr_e = librosa.load(wav_e, sr=SR)
        #     x_o, sr_o = librosa.load(wav_o, sr=SR)

        #     cep_list_a = get_cepstrum(x_a)
        #     cep_list_i = get_cepstrum(x_i)
        #     cep_list_u = get_cepstrum(x_u)
        #     cep_list_e = get_cepstrum(x_e)
        #     cep_list_o = get_cepstrum(x_o)

        #     mu_a = np.mean(np.array(cep_list_a), axis=0)
        #     mu_i = np.mean(np.array(cep_list_i), axis=0)
        #     mu_u = np.mean(np.array(cep_list_u), axis=0)
        #     mu_e = np.mean(np.array(cep_list_e), axis=0)
        #     mu_o = np.mean(np.array(cep_list_o), axis=0)

        #     var_a = np.var(np.array(cep_list_a), axis=0)
        #     var_i = np.var(np.array(cep_list_i), axis=0)
        #     var_u = np.var(np.array(cep_list_u), axis=0)
        #     var_e = np.var(np.array(cep_list_e), axis=0)
        #     var_o = np.var(np.array(cep_list_o), axis=0)

        #     return mu_a, mu_i, mu_u, mu_e, mu_o, var_a, var_i, var_u, var_e, var_o

        # mu_a, mu_i, mu_u, mu_e, mu_o, var_a, var_i, var_u, var_e, var_o = vowel_recognition(x)
        # スペクトログラムを保存するlist
        spectrogram = []
        hz_list = []
        pred = []
        # autocorr = np.correlate(x, x, 'full')

        # 不要な前半を捨てる
        # autocorr = autocorr[len(autocorr) // 2:] 
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
            # autocorr_interval = autocorr[start_idx:end_idx]
                # ピークのインデックスを抽出する
            # peakindices = [i for i in range(len(autocorr_interval)) if is_peak(autocorr_interval, i)]
            #     # インデックス0 がピークに含まれていれば捨てる
            # peakindices = [i for i in peakindices if i != 0]
            # # 自己相関が最大となるインデックスを得る
            # if peakindices:
            #     max_peak_index = max(peakindices, key=lambda index: autocorr_interval[index])
            # max_peak_index_interval = np.argmax(autocorr_interval)
            # 区間ごとの周波数を計算して出力
            # freq_interval = SR / max_peak_index
            # hz_list.append(freq_interval)

            # 母音の判定
            # cep = np.real(np.fft.rfft(fft_log_abs_spec))
            # cep = cep[:13]
            # likelihood_a = calc_likelihood(cep, mu_a, var_a)
            # likelihood_i = calc_likelihood(cep, mu_i, var_i)
            # likelihood_u = calc_likelihood(cep, mu_u, var_u)
            # likelihood_e = calc_likelihood(cep, mu_e, var_e)
            # likelihood_o = calc_likelihood(cep, mu_o, var_o)
            # likelihood = [likelihood_a, likelihood_i, likelihood_u, likelihood_e, likelihood_o]
            # pred.append((likelihood.index(max(likelihood))+ 1)* SR / 10)

        return spectrogram, hz_list, pred
     
    print("open5")
    global spectrogram
    spectrogram, hz_list, pred = calc(x)
    print("open6")

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
        # x_data = np.linspace(0, duration, len(hz_list))
        # ax3.plot(x_data, hz_list, c='b')
        # ax3.plot(x_data, pred, c='r')
        # time_axis_pred = np.linspace(0, len(pred) * size_shift, num=len(pred))
        # plt.plot(time_axis_pred, pred, color="red")
        canvas.get_tk_widget().pack(side="left")    # 最後にFrameに追加する処理

    _draw_data(spectrogram, hz_list, pred)
    print("open7")

    # スペクトルを表示する領域を確保
    # ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
    global fig2, ax2, canvas2
    fig2, ax2 = plt.subplots()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.get_tk_widget().pack(side="top")    # "top"は上部方向にウィジェットを積むことを意味する

    print("open8")
    # スライドバーを作成
    global scale
    scale = tkinter.Scale(
        # command=draw_spectrum,        # スライドバーの値を取得するメソッド
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

# スライドバーの値を更新する関数
def update_slider_position():
    # スライドバーの値を再生位置に合わせる
    scale.set(now_playing_sec)

# スライドバーの値が変更されたときに呼び出されるコールバック関数
# ここで右側のグラフに
# vはスライドバーの値
def draw_spectrum(v, spectrogram, ax2, canvas2, duration):

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


# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため

def set_music():
    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
    global is_gui_running, now_playing_sec, x, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music

    # トレモロの係数をかける
    # audio_data = audio_data * (1 + D * np.sin(2 * np.pi * R * np.arange(len(audio_data)) / SR))

    # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
    # audio_changed = (audio_changed * 32768.0). astype('int16')
    p_play = pyaudio.PyAudio()
    stream_play = p_play.open(
        format=p_play.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
        channels=audio_data.channels,								# チャネル数
        rate=audio_data.frame_rate,								# サンプリングレート
        output=True												# 出力モードに設定
    )
    # pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
    # 第二引数には何ミリ秒毎に読み込むかを指定
    # ここでは10ミリ秒ごとに読み込む

    size_frame_music = 10  # 10ミリ秒毎に読み込む
    idx = 0
    D = float(d.get())
    R = float(r.get())
    # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
    #
    # なぜ再生するだけのためにフレーム毎の処理をするのか？
    # 音楽ファイルに対しても何らかの処理を行えるようにするため（このサンプルプログラムでは行っていない）
    # おまけに，再生位置も計算することができる
    for chunk in make_chunks(audio_data, size_frame_music):

        # GUIが終了してれば，この関数の処理も終了する
        if is_gui_running == False:
            break

        # データの取得
        data_music = np.array(chunk.get_array_of_samples())
        # type(chunk.get_array_of_samples())
        # 正規化
        data_music_changed = data_music * (1 + D * np.sin(2 * np.pi * R * np.arange(len(data_music)) / SR))
        chunk = make_chunks(data_music_changed, size_frame_music)

        # pyaudioの再生ストリームに切り出した音楽データを流し込む
        # 再生が完了するまで処理はここでブロックされる
        stream_play.write(chunk._data)
        
        # 現在の再生位置を計算（単位は秒）
        now_playing_sec = (idx * size_frame_music) / 1000.
        
        idx += 1

        #
        # 【補足】
        # 楽曲のスペクトログラムを計算する場合には下記のように楽曲のデータを受け取る
        # ただし，音声データの値は -1.0~1.0 ではなく，16bit の整数値であるので正規化を施している
        # また十分なサイズの音声データを確保してからfftを実行すること
        # 楽曲が44.1kHzの場合，44100 / (1000/10) のサイズのデータとなる
        # 以下では処理のみを行い，表示はしない．表示をするには animate 関数の
        # 中身を変更すること 
        data_music = data_music / np.iinfo(np.int32).max   
        #
        # 以下はマイク入力のときと同様
        #

        # 現在のフレームとこれまでに入力されたフレームを連結
        x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

        # フレームサイズ分のデータがあれば処理を行う
        # if len(x_stacked_data_music) >= size_frame:
        #     # フレームサイズからはみ出した過去のデータは捨てる
        #     x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)-size_frame:]

        #     # スペクトルを計算
        #     fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
        #     fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

        #     # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
        #     # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
        #     spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
        #     spectrogram_data_music[:, -1] = fft_log_abs_spec
        # update_slider_position()


# 再生時間の表示を随時更新する関数
def update_gui_text():
    global is_gui_running, now_playing_sec

    while is_gui_running:
        # 0.01秒ごとに更新
        draw_spectrum(now_playing_sec, spectrogram, ax2, canvas2, duration)
        update_slider_position()
        time.sleep(0.01)

def play_music(event):
    # 再生時間を表す
    global now_playing_sec
    now_playing_sec = 0.0
    global is_playing, is_gui_running, t_play_music
    is_gui_running = True
    print("play")
    t_play_music = threading.Thread(target=set_music)
    print("play2")
    t_play_music.start()
    t_update_gui = threading.Thread(target=update_gui_text)
    t_update_gui.start()


# Tkinterを初期化
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-SAMPLE")
d = tkinter.StringVar(root)
r = tkinter.StringVar(root)


# ファイルを開く
open_file_button = tkinter.Button(root, text='Open file')
open_file_button.pack(side=tkinter.TOP)
open_file_button.bind('<Button-1>', open_file)

# 再生ボタン
play_music_button = tkinter.Button(root, text='Play music')
play_music_button.pack(side=tkinter.BOTTOM)
play_music_button.bind('<Button-1>', play_music)

# トレモロの係数を変更する
# Scaleで設定
# tkinterでスライドバーを作成する

# Labelの生成
label_d = tkinter.Label(
    root,
    textvariable=d,   #varを表示
    relief="ridge",
    )
label_d.pack()

#Scaleの生成
scale_d = tkinter.Scale(
    root,
    variable=d,   #スケールの値をvarにセット
    )
scale_d.pack()

# Labelの生成
label_r = tkinter.Label(
    root,
    textvariable=r,   #varを表示
    relief="ridge",
    )
label_r.pack()

#Scaleの生成
scale_r = tkinter.Scale(
    root,
    variable=r,   #スケールの値をvarにセット
    )
scale_r.pack()





tkinter.mainloop()