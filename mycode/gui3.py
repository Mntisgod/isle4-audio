# ライブラリの読み込み
import pyaudio
import numpy as np
import threading
import time
import math

# matplotlib関連＼
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from tkmacosx import Button
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 4096    # 今回は0.256秒

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE / 20)    # 今回は0.05秒

# スペクトルをカラー表示する際に色の範囲を正規化するために
# スペクトルの最小値と最大値を指定
# スペクトルの値がこの範囲を超えると，同じ色になってしまう
SPECTRUM_MIN = -30
SPECTRUM_MAX = 15

# 音量を表示する際の値の範囲
VOLUME_MIN = -120
VOLUME_MAX = -10

# log10を計算する際に，引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100

# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False

#
# (1) GUI / グラフ描画の処理
#

# ここでは matplotlib animation を用いて描画する
# 毎回 figure や ax を初期化すると処理時間がかかるため
# データを更新したら，それに従って必要な部分のみ再描画することでリアルタイム処理を実現する

# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う

# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
    if frequency == 0:
        return 0
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69

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


def light_up_button(nn):
    btn = btn_list[nn]
    # btnの色を変える
    print(nn)
    # btn.update()
    btn['bg'] = 'red'
    btn.place()

    if nn % 12 == 0 or nn % 12 == 2 or nn % 12 == 4 or nn % 12 == 5 or nn % 12 == 7 or nn % 12 == 9 or nn % 12 == 11:
        btn.place()
        root.after(500, lambda: change_color(btn, 'white'))
        btn.place()
    else:
        btn.place()
        root.after(100, lambda: change_color(btn, 'black'))
        print("hoge")
        btn.place()

def change_color(btn, color):
    btn['bg'] = color

def calculate_pitch(signal, vol):
    """
    Calculate pitch (fundamental frequency) using autocorrelation.

    Parameters:
    - signal: numpy array
        Input audio signal.
    - sampling_rate: int
        Sampling rate of the audio signal.

    Returns:
    - pitch: float
        Estimated fundamental frequency in Hertz.
    """

    # Autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')

    # Take only the positive part (since we are interested in positive lags)
    autocorr = autocorr[len(autocorr)//2:]

    # Find the index of the maximum value (excluding the first peak, which is the autocorrelation with itself)
    # peak_index = np.argmax(autocorr[1:]) + 1

    peakindices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]
    # インデックス0 がピークに含まれていれば捨てる
    peakindices = [i for i in peakindices if i != 0]
    # 自己相関が最大となるインデックスを得る
    if peakindices:
        max_peak_index = max(peakindices, key=lambda index: autocorr[index])
        freq_interval = SAMPLING_RATE / max_peak_index
    else:
        max_peak_index = 0
        freq_interval = 0
    if freq_interval > 550 or vol < -100:
        freq_interval = 0
    return freq_interval


def is_peak(a, index):
    if index == 0 or index == len(a)-1:
        return False
    if a[index-1] < a[index] and a[index] > a[index+1]:
        return True
    else:
        return False


def animate(frame_index):

    # ax1_sub.set_array(spectrogram_data)

    # この上の処理を下記のようにすれば楽曲のスペクトログラムが表示される
    ax1_sub.set_array(spectrogram_data_music * 4)
    ax2_sub.set_data(time_x_data, pitch_data)
    plt.yticks(note_freq, note_name)
    return ax1_sub, ax2_sub

# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-SAMPLE")

# スペクトログラムを描画
fig, ax1 = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=root)

# 横軸の値のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
# 縦軸の値のデータ
freq_y_data = np.linspace(10, 1000, MAX_NUM_SPECTROGRAM)

# とりあえず初期値（ゼロ）のスペクトログラムと音量のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
volume_data = np.zeros(len(time_x_data))
# str型で音程を格納するデータ
pitch_data = np.zeros(len(time_x_data))
music_pitch_data = np.zeros(len(time_x_data))

# 楽曲のスペクトログラムを格納するデータ（このサンプルでは計算のみ）
spectrogram_data_music = np.zeros((len(freq_y_data), len(time_x_data)))

# スペクトログラムを描画する際に横軸と縦軸のデータを行列にしておく必要がある
# これは下記の matplotlib の pcolormesh の仕様のため
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
    for idx_t, t_v in enumerate(time_x_data):
        X[idx_f, idx_t] = t_v
        Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub = ax1.pcolormesh(
    X,
    Y,
    spectrogram_data,
    shading='nearest',    # 描画スタイル
    cmap='plasma',            # カラーマップ
    norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)    # 値の最小値と最大値を指定して，それに色を合わせる
)
ax1.set_yscale('log')

note_freq = [440 * 2 ** ((i - 69) / 12) for i in range(24, 73, 12)]
note_name = [nn2key(i) for i in range(24, 73, 12)]
print(note_name)
print(note_freq)
# 入力波形の基本周波数を表示するため軸を作成
ax2 = ax1.twinx()
plt.yticks(note_freq, note_name)
ax2.set_yscale('log')
ax2.set_ylim(10, 1000)
# 基本周波数をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax2_sub, = ax2.plot(time_x_data, pitch_data, c='r')

# 同じ軸に楽曲の基本周波数を表示する
# ax2_sub2 = ax2.plot(time_x_data, music_pitch_data, c='black')

# ラベルの設定
ax1.set_xlabel('sec')                # x軸のラベルを設定
ax1.set_ylabel('frequency [Hz]')    # y軸のラベルを設定

# 音量を表示する際の値の範囲を設定

# maplotlib animationを設定
ani = animation.FuncAnimation(
    fig,
    animate,        # 再描画のために呼び出される関数
    interval=100,    # 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
    blit=True        # blitting処理を行うため描画処理が速くなる
)

# matplotlib を GUI(Tkinter) に追加する
toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master=root, textvariable=text, font=("", 30))
label.pack()

pitch_text = tkinter.StringVar()
pitch_text.set('')
label = tkinter.Label(master=root, textvariable=pitch_text, font=("", 30))
label.pack()
# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
    root.quit()
    root.destroy()

# 終了ボタンを作成
button = tkinter.Button(master=root, text="終了", command=_quit, font=("", 30))
button.pack()


#
# (2) マイク入力のための処理
#

x_stacked_data = np.array([])

# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):
    
    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの numpy array を利用できるように global 宣言する
    # これらにはフレーム毎のスペクトルと音量のデータが格納される
    global x_stacked_data, spectrogram_data, volume_data, pitch_data

    # 現在のフレームの音声データをnumpy arrayに変換
    x_current_frame = np.frombuffer(in_data, dtype=np.float32)

    # 現在のフレームとこれまでに入力されたフレームを連結
    x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

    # フレームサイズ分のデータがあれば処理を行う
    if len(x_stacked_data) >= FRAME_SIZE:
        
        # フレームサイズからはみ出した過去のデータは捨てる
        x_stacked_data = x_stacked_data[len(x_stacked_data)-FRAME_SIZE:]

        # スペクトルを計算
        fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
        fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

        # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
        # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = fft_log_abs_spec

        # 音量も同様の処理
        vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
        volume_data = np.roll(volume_data, -1)
        volume_data[-1] = vol

        # 基本周波数を計算
        # ここに基本周波数を計算する処理を書く
        # ただし，基本周波数を計算する関数は別途作成すること
        pitch = calculate_pitch(x_stacked_data, vol)
        pitch_data = np.roll(pitch_data, -1)
        pitch_data[-1] = pitch
        nn = hz2nn(pitch)
        key = nn2key(nn)
        pitch_text.set(key)
        if key != '0':
        # 別スレッドで実行するため，GUIの更新は root.after を用いる
            light_up_button(nn-24)
            # root.after(0, light_up_button, nn-24)
        # btn_list[nn-24].invoke()


        # print()
    
    # 戻り値は pyaudio の仕様に従うこと
    return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
# 【注意】シフトサイズごとに指定された関数が呼び出される
p = pyaudio.PyAudio()
stream = p.open(
    format = pyaudio.paFloat32,
    channels = 1,
    rate = SAMPLING_RATE,
    input = True,                        # ここをTrueにするとマイクからの入力になる 
    frames_per_buffer = SHIFT_SIZE,        # シフトサイズ
    stream_callback = input_callback    # ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) mp3ファイル音楽を再生する処理
#

# mp3ファイル名
# ここは各自の音源ファイルに合わせて変更すること
filename = './shs-test-midi.wav'

#
# 【注意】なるべく1チャネルの音声を利用すること
# ステレオ（2チャネル）の場合は SoX などでモノラルに変換できる
# sox ソ連国歌.mp3 -c 1 monosoviet.wav
#

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_file(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
    format = p.get_format_from_width(audio_data.sample_width),    # ストリームを読み書きするときのデータ型
    channels = audio_data.channels,                                # チャネル数
    rate = audio_data.frame_rate,                                # サンプリングレート
    output = True                                                # 出力モードに設定
)

# 楽曲のデータを格納
x_stacked_data_music = np.array([])


# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def play_music():

    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
    global is_gui_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music

    # pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
    # 第二引数には何ミリ秒毎に読み込むかを指定
    # ここでは10ミリ秒ごとに読み込む

    size_frame_music = 10    # 10ミリ秒毎に読み込む

    idx = 0

    # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
    #
    # なぜ再生するだけのためにフレーム毎の処理をするのか？
    # 音楽ファイルに対しても何らかの処理を行えるようにするため（このサンプルプログラムでは行っていない）
    # おまけに，再生位置も計算することができる
    for chunk in make_chunks(audio_data, size_frame_music):
        
        # GUIが終了してれば，この関数の処理も終了する
        if is_gui_running == False:
            break

        # pyaudioの再生ストリームに切り出した音楽データを流し込む
        # 現在の再生位置を計算（単位は秒）
        now_playing_sec = (idx * size_frame_music) / 1000.
        
        idx += 1

        #
        # 【補足】
        # 楽曲のスペクトログラムを計算する場合には下記のように楽曲のデータを受け取る
        # ただし，音声データの値は -1.0~1.0 ではなく，16bit の整数値であるので正規化を施している
        # また十分なサイズの音声データを確保してからfftを実行すること
        # 楽曲が44.1kHzの場合，44100 / (1000/10) のサイズのデータとなる
        # 以下では処理のみを行い，表示はしない．表示をするには animate 関数の中身を変更すること 
        
        # データの取得
        data_music = np.array(chunk.get_array_of_samples())
        
        # 正規化
        data_music = data_music / np.iinfo(np.int32).max    

        
        #
        # 以下はマイク入力のときと同様
        #

        # 現在のフレームとこれまでに入力されたフレームを連結
        x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

        # フレームサイズ分のデータがあれば処理を行う
        if len(x_stacked_data_music) >= FRAME_SIZE:
            
            # フレームサイズからはみ出した過去のデータは捨てる
            x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)-FRAME_SIZE:]

            # スペクトルを計算
            fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window) * 2000
            fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

            # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
            # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
            spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
            spectrogram_data_music[:, -1] = fft_log_abs_spec

            # threading.Thread(target=calculate_pitch, args=(x_stacked_data_music, 0)).start()
            # pitch = calculate_pitch(x_stacked_data_music, 0)
            # music_pitch_data = np.roll(pitch_data, -1)
            # music_pitch_data[-1] = pitch
        # 再生が完了するまで処理はここでブロックされる
        stream_play.write(chunk._data)
        

# 再生時間の表示を随時更新する関数
def update_gui_text():

    global is_gui_running, now_playing_sec, text

    while True:
        
        # GUIが表示されていれば再生位置（秒）をテキストとしてGUI上に表示
        if is_gui_running:
            text.set('%.3f' % now_playing_sec)
        
        # 0.01秒ごとに更新
        time.sleep(0.01)

# 再生時間を表す
now_playing_sec = 0.0

# 音楽を再生するパートを関数化したので，それを別スレッドで（GUIのため）再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)    # GUIが消されたときにこの別スレッドの処理も終了されるようにするため

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)    # GUIが消されたときにこの別スレッドの処理も終了されるようにするため

#
# (4) 全体の処理を実行
#

# GUIの開始フラグをTrueに
is_gui_running = True

# 上記で設定したスレッドを開始（直前のフラグを立ててから）
t_play_music.start()
t_update_gui.start()

# frame_v = tkinter.Frame(
#         root,
#         relief="ridge",
#         borderwidth=20,
#         )
# frame_v.pack(side=tkinter.BOTTOM)
btn01 = Button(root, text='', bg='white', command=lambda:light_up_button(0))
btn01.place(x=190, y=30, height=100, width=30)
btn02 = Button(root, bg='black', command=lambda:light_up_button(1))
btn02.place(x=210, y=30, height=60, width=20)
btn03 = Button(root, text='', bg='white', command=lambda:light_up_button(2))
btn03.place(x=220, y=30, height=100, width=30)
btn04 = Button(root, bg='black', command=lambda:light_up_button(3))
btn04.place(x=240, y=30, height=60, width=20)
btn05 = Button(root, text='', bg='white', command=lambda:light_up_button(4))
btn05.place(x=250, y=30, height=100, width=30)
btn06 = Button(root, text='', bg='white', command=lambda:light_up_button(5))
btn06.place(x=280, y=30, height=100, width=30)
btn07 = Button(root, bg='black', command=lambda:light_up_button(6))
btn07.place(x=300, y=30, height=60, width=20)
btn08 = Button(root, text='', bg='white', command=lambda:light_up_button(7))
btn08.place(x=310, y=30, height=100, width=30)
btn09 = Button(root, bg='black', command=lambda:light_up_button(8))
btn09.place(x=330, y=30, height=60, width=20)
btn10 = Button(root, text='', bg='white', command=lambda:light_up_button(9))
btn10.place(x=340, y=30, height=100, width=30)
btn11 = Button(root, bg='black', command=lambda:light_up_button(10))
btn11.place(x=360, y=30, height=60, width=20)
btn12 = Button(root, text='', bg='white', command=lambda:light_up_button(11))
btn12.place(x=370, y=30, height=100, width=30)


btn13 = Button(root, text='', bg='white', command= lambda:light_up_button(12))
btn13.place(x=400, y=30, height=100, width=30)
btn14 = Button(root, bg='black', command=lambda:light_up_button(13))
btn14.place(x=420, y=30, height=60, width=20)
btn15 = Button(root, text='', bg='white', command=lambda:light_up_button(14))
btn15.place(x=430, y=30, height=100, width=30)
btn16 = Button(root, bg='black', command=lambda:light_up_button(15))
btn16.place(x=450, y=30, height=60, width=20)
btn17 = Button(root, text='', bg='white', command=lambda:light_up_button(16))
btn17.place(x=460, y=30, height=100, width=30)
btn18 = Button(root, text='', bg='white', command=lambda:light_up_button(17))
btn18.place(x=490, y=30, height=100, width=30)
btn19 = Button(root, bg='black', command=lambda:light_up_button(18))
btn19.place(x=510, y=30, height=60, width=20)
btn20 = Button(root, text='', bg='white', command=lambda:light_up_button(19))
btn20.place(x=520, y=30, height=100, width=30)
btn21 = Button(root, bg='black', command=lambda:light_up_button(20))
btn21.place(x=540, y=30, height=60, width=20)
btn22 = Button(root, text='', bg='white', command=lambda:light_up_button(21))
btn22.place(x=550, y=30, height=100, width=30)
btn23 = Button(root, bg='black',fg='purple' , command=lambda:light_up_button(22))
btn23.place(x=570, y=30, height=60, width=20)
btn24 = Button(root, text='', bg='white', command=lambda:light_up_button(23))
btn24.place(x=580, y=30, height=100, width=30)

btn25 = Button(root, text='', bg='white', command=lambda:light_up_button(24))
btn25.place(x=610, y=30, height=100, width=30)
btn26 = Button(root, bg='black', command=lambda:light_up_button(25))
btn26.place(x=630, y=30, height=60, width=20)
btn27 = Button(root, text='', bg='white', command=lambda:light_up_button(26))
btn27.place(x=640, y=30, height=100, width=30)
btn28 = Button(root, bg='black', command=lambda:light_up_button(27))
btn28.place(x=660, y=30, height=60, width=20)
btn29 = Button(root, text='', bg='white', command=lambda:light_up_button(28))
btn29.place(x=670, y=30, height=100, width=30)
btn30 = Button(root, text='', bg='white', command=lambda:light_up_button(29))
btn30.place(x=700, y=30, height=100, width=30)
btn31 = Button(root, bg='black', command=lambda:light_up_button(30))
btn31.place(x=720, y=30, height=60, width=20)
btn32 = Button(root, text='', bg='white', command=lambda:light_up_button(31))
btn32.place(x=730, y=30, height=100, width=30)
btn33 = Button(root, bg='black', command=lambda:light_up_button(32))
btn33.place(x=750, y=30, height=60, width=20)
btn34 = Button(root, text='', bg='white', command=lambda:light_up_button(33))
btn34.place(x=760, y=30, height=100, width=30)
btn35 = Button(root, bg='black', command=lambda:light_up_button(34))
btn35.place(x=780, y=30, height=60, width=20)
btn36 = Button(root, text='', bg='white', command=lambda:light_up_button(35))
btn36.place(x=790, y=30, height=100, width=30)

btn37 = Button(root, text='', bg='white', command=lambda:light_up_button(36))
btn37.place(x=820, y=30, height=100, width=30)

btn38 = Button(root, bg='black', command=lambda:light_up_button(37))
btn38.place(x=840, y=30, height=60, width=20)
btn39 = Button(root, text='', bg='white',command=lambda:light_up_button(38))
btn39.place(x=850, y=30, height=100, width=30)
btn40 = Button(root, bg='black', command=lambda:light_up_button(39))
btn40.place(x=870, y=30, height=60, width=20)
btn41 = Button(root, text='', bg='white', command=lambda:light_up_button(40))
btn41.place(x=880, y=30, height=100, width=30)
btn42 = Button(root, text='', bg='white', command=lambda:light_up_button(41))
btn42.place(x=910, y=30, height=100, width=30)
btn43 = Button(root, bg='black', command=lambda:light_up_button(42))
btn43.place(x=930, y=30, height=60, width=20)
btn44 = Button(root, text='', bg='white', command=lambda:light_up_button(43))
btn44.place(x=940, y=30, height=100, width=30)
btn45 = Button(root, bg='black', command=lambda:light_up_button(44))
btn45.place(x=960, y=30, height=60, width=20)
btn46 = Button(root, text='', bg='white', command=lambda:light_up_button(45))
btn46.place(x=970, y=30, height=100, width=30)
btn47 = Button(root, bg='black', command=lambda:light_up_button(46))
btn47.place(x=990, y=30, height=60, width=20)
btn48 = Button(root, text='', bg='white', command=lambda:light_up_button(47))
btn48.place(x=1000, y=30, height=100, width=30)

btn49 = Button(root, text='', bg='white', command=lambda:light_up_button(48))
btn49.place(x=1030, y=30, height=100, width=30)


btn_list = [btn01, btn02, btn03, btn04, btn05, btn06, btn07, btn08, btn09,
            btn10, btn11, btn12, btn13, btn14, btn15, btn16, btn17, btn18, btn19,
            btn20, btn21, btn22, btn23, btn24, btn25, btn26, btn27, btn28, btn29,
            btn30, btn31, btn32, btn33, btn34, btn35, btn36, btn37, btn38, btn39,
            btn40, btn41, btn42, btn43, btn44, btn45, btn46, btn47, btn48, btn49]

# GUIを開始，GUIが表示されている間は処理はここでストップ
tkinter.mainloop()

# GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
is_gui_running = False

# 終了処理
stream_play.stop_stream()
stream_play.close()
p_play.terminate()