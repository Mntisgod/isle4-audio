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
SR = 16000     # サンプリングレート
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
    # wavかmp3ファイルを選択する
    fTyp = [("wav file", "*.wav"), ("mp3 file", "*.mp3")]
    # fTyp = [("wav file", "*.wav")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    global input_file
    print("open")
    input_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    # 音声ファイルを読み込む
    global x
    x, SR = librosa.load(input_file, sr=None)
    global audio_data
    audio_data = AudioSegment.from_file(input_file)
    audio_data = audio_data.set_channels(1)
    audio_data = audio_data.set_sample_width(2)
    global duration
    # ファイルサイズ（秒）
    duration = len(x) / SR

    def calc(x):
        # スペクトログラムを保存するlist
        spectrogram = []

        for i in np.arange(0, len(x)-size_frame, size_shift):
            
            # 該当フレームのデータを取得
            start_idx = int(i)    # arangeのインデクスはfloatなのでintに変換
            end_idx = start_idx+size_frame
            x_frame = x[start_idx: end_idx]

            # スペクトル
            fft_spec = np.fft.rfft(x_frame * hamming_window)
            fft_log_abs_spec = np.log(np.abs(fft_spec))
            spectrogram.append(fft_log_abs_spec)

        return spectrogram

    global spectrogram
    spectrogram = calc(x)

    # Tkinterのウィジェットを階層的に管理するためにFrameを使用
    # frame1 ... スペクトログラムを表示
    # frame2 ... Scale（スライドバー）とスペクトルを表示
    frame1 = tkinter.Frame(root)
    frame2 = tkinter.Frame(root)
    frame1.pack(side="left")
    frame2.pack(side="left")

    def _draw_data(spectrogram):
        # まずはスペクトログラムを描画
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=frame1)    # masterに対象とするframeを指定
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('sec')
        ax1.set_ylabel('frequency [Hz]')
        ax1.imshow(
            np.flipud(np.array(spectrogram)),
            extent=[0, duration, 0, 8000],
            aspect='auto',
            cmap='inferno',
            interpolation='nearest'
        )
        # 続いて右側のy軸を追加して，音量を重ねて描画
        ax3 = ax1.twinx()
        ax3.set_ylabel('frequency [Hz]')
        canvas.get_tk_widget().pack(side="left")    # 最後にFrameに追加する処理

    _draw_data(spectrogram)

    # スペクトルを表示する領域を確保
    # ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
    global fig2, ax2, canvas2
    fig2, ax2 = plt.subplots()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.get_tk_widget().pack(side="top")    # "top"は上部方向にウィジェットを積むことを意味する

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
def draw_spectrum(t, spectrogram, ax2, canvas2, duration):

    # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
    index = int((len(spectrogram)-1) * (float(t) / duration))
    spectrum = spectrogram[index]

    # 直前のスペクトル描画を削除し，新たなスペクトルを描画
    plt.cla()
    x_data = np.fft.rfftfreq(size_frame, d=1/SR)
    ax2.plot(x_data, spectrum)
    ax2.set_ylim(-10, 5)
    ax2.set_xscale('log')
    ax2.set_xlim(10, SR/2)
    ax2.set_ylabel('amblitude')
    ax2.set_xlabel('frequency [Hz]')

    canvas2.draw()


# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def set_music():
    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
    global is_gui_running, now_playing_sec, audio_data, now_playing_sec, x_stacked_data_music

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
    freq = float(f.get())
    amplitude = float(amp.get())
    is_voice_change = var.get()

    # ピッチを変える
    # audio_data = audio_data.set_frame_rate(int(audio_data.frame_rate * (1 + D * np.sin(2 * np.pi * R * np.arange(len(audio_data)) / SR))))
    # audio_data = audio_data * 2 ** (1)
    # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
    # なぜ再生するだけのためにフレーム毎の処理をするのか？
    # 音楽ファイルに対しても何らかの処理を行えるようにするため（このサンプルプログラムでは行っていない）
    # おまけに，再生位置も計算することができる
    for chunk in make_chunks(audio_data, size_frame_music):

        # GUIが終了してれば，この関数の処理も終了する
        if is_gui_running == False:
            break
        
        print(chunk.sample_width)
        print(chunk.frame_rate)
        print(chunk.channels)
        print(chunk._data)
        # データの取得
        data_music = np.array(chunk.get_array_of_samples())        
        data = data_music * (1 + D * np.sin(2 * np.pi * R * np.arange(len(data_music)) / SR))
        data = data.astype('int16')

        # 正弦波を生成する関数
        # sampling_rate ... サンプリングレート
        # frequency ... 生成する正弦波の周波数
        # duration ... 生成する正弦波の時間的長さ
        def generate_sinusoid(sampling_rate, frequency, duration):
            sampling_interval = 1.0 / sampling_rate
            t = np.arange(sampling_rate * duration) * sampling_interval
            waveform = np.sin(2.0 * np.pi * frequency * t)
            return waveform

        # 元の音声と正弦波を重ね合わせる
        if is_voice_change:
            # 生成する正弦波の周波数（Hz）
            frequency = freq

            # 生成する正弦波の時間的長さ
            duration = len(data_music)

            # 正弦波を生成する
            sin_wave = generate_sinusoid(SR, frequency, duration/SR)

            # 最大値を指定
            sin_wave = sin_wave * amplitude
            data = data * sin_wave
            data = data.astype('int16')

        sound = AudioSegment(data=b''.join(data),
                             sample_width=2,
                             frame_rate=44100,
                             channels=1
                             )
        # pyaudioの再生ストリームに切り出した音楽データを流し込む
        # 再生が完了するまで処理はここでブロックされる
        stream_play.write(sound._data)

        # 現在の再生位置を計算（単位は秒）
        now_playing_sec = (idx * size_frame_music) / 1000.
        idx += 1


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


if __name__ == '__main__':
    # Tkinterを初期化
    root = tkinter.Tk()
    root.wm_title("EXP4-AUDIO-SAMPLE")

    # トレモロの係数を変更する
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

    # ボイスチェンジャの係数を変更する
    f = tkinter.StringVar(root)
    amp = tkinter.StringVar(root)
    # トレモロのframe
    frame_t = tkinter.Frame(
        root,
        relief="ridge",
        borderwidth=20, 
        )
    frame_t.pack(side=tkinter.LEFT)
    # Labelの生成
    label_d = tkinter.Label(
        frame_t,
        textvariable=d,   #varを表示
        relief="ridge",
        text="トレモロの深さ"
        )
    label_d.pack()

    #Scaleの生成
    scale_d = tkinter.Scale(
        frame_t,
        variable=d,   #スケールの値をvarにセット
        from_=0,                    # 最小値
        to=10,                # 最大値
        width=10,
        resolution=0.1,    # 刻み幅
        )
    scale_d.pack()

    # Labelの生成
    label_r = tkinter.Label(
        frame_t,
        textvariable=r,   #varを表示
        relief="ridge",
        text="トレモロの速さ"
        )
    label_r.pack()

    #Scaleの生成
    scale_r = tkinter.Scale(
        frame_t,
        variable=r,   #スケールの値をvarにセット
        from_=0,                    # 最小値
        to=10,                # 最大値
        resolution=0.1,    # 刻み幅
        width=10,
        )
    scale_r.pack()

    # ボイスチェンジャの係数を変更する
    # ボイスチェンジャのframe
    frame_v = tkinter.Frame(
        root,
        relief="ridge",
        borderwidth=20,
        )
    frame_v.pack(side=tkinter.LEFT)
    # Labelの生成

    label_f = tkinter.Label(
        frame_v,
        textvariable=f,
        relief="ridge",
        text="ボイスチェンジャにかける正弦波の周波数"
        )
    label_f.pack()

    #Scaleの生成
    scale_f = tkinter.Scale(
        frame_v,
        variable=f,
        from_=0,  # 最小値
        to=1000,  # 最大値
        width=10,
        resolution=1,  # 刻み幅
        )
    scale_f.pack()

    label_amp = tkinter.Label(
        frame_v,
        textvariable=amp,
        relief="ridge",
        text="ボイスチェンジャにかける正弦波の周波数"
        )
    label_amp.pack()

    #Scaleの生成
    scale_amp = tkinter.Scale(
        frame_v,
        variable=amp,  # スケールの値をvarにセット
        from_=0,                    # 最小値
        to=1,                # 最大値
        width=10,
        resolution=0.01,    # 刻み幅
        # orient=tkinter.HORIZONTAL,    # 横方向にスライド
        )
    scale_amp.pack()

    # ボイスチェンジをするかどうか
    # checkbuttonの生成
    var = tkinter.BooleanVar()
    var.set(False)
    c = tkinter.Checkbutton(
        root,
        variable=var,
        text="use voice changer"
        )
    c.pack(side=tkinter.LEFT)

    tkinter.mainloop()