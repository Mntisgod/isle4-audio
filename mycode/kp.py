import tkinter as tk1 
import threading as th1 
import time as tm1 
import pygame.midi

class tone_thread1( th1.Thread ): 
    def __init__( self ): 
        super( tone_thread1, self ).__init__()
    def run( self ):
        tone1() 

def tone1(): 
    global n1 
    global vol1 
    out1.note_on( n1, vol1 )
    tm1.sleep( 2.0 ) 
    out1.note_off( n1, vol1 ) 

def btn1( n0 ): 
    global n1 
    n1 = n0 - 25 + 60 
    thr1 = tone_thread1() 
    thr1.start() 

def key1( event ): 
    global n1 
    c1 = event.char 
    n0 = 0 
    if c1 == 'a': 
        n0 = 60
    elif c1 == 'w': 
        n0 = 61 
    elif c1 == 's': 
        n0 = 62 
    elif c1 == 'e': 
        n0 = 63 
    elif c1 == 'd': 
        n0 = 64 
    elif c1 == 'f': 
        n0 = 65 
    elif c1 == 'y': 
        n0 = 66
    elif c1 == 'g': 
        n0 = 67 
    elif c1 == 'y': 
        n0 = 68 
    elif c1 == 'h': 
        n0 = 69 
    elif c1 == 'u': 
        n0 = 70 
    elif c1 == 'j': 
        n0 = 71 
    elif c1 == 'k': 
        n0 = 72 
    elif c1 == 'o': 
        n0 = 73 
    elif c1 == 'l': 
        n0 = 74 
    elif c1 == 'p': 
        n0 = 75 
    elif c1 == ';': 
        n0 = 76 
    elif c1 == ':': 
        n0 = 77 
    elif c1 == '[': 
        n0 = 78
    elif c1 == ']': 
        n0 = 79
    if n0 > 0: 
        n1 = n0 
        thr1 = tone_thread1()
        thr1.start() 

def inst1( n1 ): 
    global ins1 
    global chn1 
    ins0 = textbox1.get() 
    if ins0.isdigit(): 
        ins0 = int( ins0 )
    else:
        ins0 = 1
    if n1 > 0: 
        ins0 = ins0 + 1 
    else:
        ins0 = ins0 - 1 
    if ins0 < 1: ins0 = 1 
    elif ins0 > 128: 
        ins0 = 128
    ins1 = ins0 - 1 
    textbox1.delete(0, tk1.END) 
    textbox1.insert(0, str(ins0) ) 
    btn_inst1.focus_set() 
    out1.set_instrument(ins1, chn1) 

def volu1( n1 ): 
    global vol1 
    vol0 = textbox2.get() 
    if vol0.isdigit(): 
        vol0 = int( vol0 )
    else:
        vol0 = 127 
    if n1 > 0: 
        vol0 = vol0 + 1 
    else:
        vol0 = vol0 - 1 
    if vol0 < 0: vol0 = 0 
    elif vol0 > 127: 
        vol0 = 127
    vol1 = vol0 
    textbox2.delete(0, tk1.END) 
    textbox2.insert(0, str(vol1) ) 
    btn_inst3.focus_set() 

root = tk1.Tk()
root.title('midi_piano v0.2')

root.geometry("910x140")
root["bg"] = "#202020" 

root.bind( '<key>', key1 ) 

btn01 = tk1.Button(root, text='', bg='white', command=lambda:btn1(1))
btn01.place(x=20, y=30, height=100, width=30)
btn02 = tk1.Button(root, text='', bg='white', command=lambda:btn1(3))
btn02.place(x=50, y=30, height=100, width=30)
btn03 = tk1.Button(root, text='', bg='white', command=lambda:btn1(5))
btn03.place(x=80, y=30, height=100, width=30)
btn04 = tk1.Button(root, text='', bg='white', command=lambda:btn1(6))
btn04.place(x=110, y=30, height=100, width=30)
btn05 = tk1.Button(root, text='', bg='white', command=lambda:btn1(8))
btn05.place(x=140, y=30, height=100, width=30)
btn06 = tk1.Button(root, text='', bg='white', command=lambda:btn1(10))
btn06.place(x=170, y=30, height=100, width=30)
btn07 = tk1.Button(root, text='', bg='white', command=lambda:btn1(12))
btn07.place(x=200, y=30, height=100, width=30)

btn08 = tk1.Button(root, text='', bg='white', command=lambda:btn1(13))
btn08.place(x=230, y=30, height=100, width=30)
btn09 = tk1.Button(root, text='', bg='white', command=lambda:btn1(15))
btn09.place(x=260, y=30, height=100, width=30)
btn10 = tk1.Button(root, text='', bg='white', command=lambda:btn1(17))
btn10.place(x=290, y=30, height=100, width=30)
btn11 = tk1.Button(root, text='', bg='white', command=lambda:btn1(18))
btn11.place(x=320, y=30, height=100, width=30)
btn12 = tk1.Button(root, text='', bg='white', command=lambda:btn1(20))
btn12.place(x=350, y=30, height=100, width=30)
btn13 = tk1.Button(root, text='', bg='white', command=lambda:btn1(22))
btn13.place(x=380, y=30, height=100, width=30)
btn14 = tk1.Button(root, text='', bg='white', command=lambda:btn1(24))
btn14.place(x=410, y=30, height=100, width=30)

btn15 = tk1.Button(root, text='', bg='white', command=lambda:btn1(25))
btn15.place(x=440, y=30, height=100, width=30)
btn16 = tk1.Button(root, text='', bg='white', command=lambda:btn1(27))
btn16.place(x=470, y=30, height=100, width=30)
btn17 = tk1.Button(root, text='', bg='white', command=lambda:btn1(29))
btn17.place(x=500, y=30, height=100, width=30)
btn18 = tk1.Button(root, text='', bg='white', command=lambda:btn1(30))
btn18.place(x=530, y=30, height=100, width=30)
btn19 = tk1.Button(root, text='', bg='white', command=lambda:btn1(32))
btn19.place(x=560, y=30, height=100, width=30)
btn20 = tk1.Button(root, text='', bg='white', command=lambda:btn1(34))
btn20.place(x=590, y=30, height=100, width=30)
btn21 = tk1.Button(root, text='', bg='white', command=lambda:btn1(36))
btn21.place(x=620, y=30, height=100, width=30)

btn22 = tk1.Button(root, text='', bg='white', command=lambda:btn1(37))
btn22.place(x=650, y=30, height=100, width=30)
btn23 = tk1.Button(root, text='', bg='white', command=lambda:btn1(39))
btn23.place(x=680, y=30, height=100, width=30)
btn24 = tk1.Button(root, text='', bg='white', command=lambda:btn1(41))
btn24.place(x=710, y=30, height=100, width=30)
btn25 = tk1.Button(root, text='', bg='white', command=lambda:btn1(42))
btn25.place(x=740, y=30, height=100, width=30)
btn26 = tk1.Button(root, text='', bg='white', command=lambda:btn1(44))
btn26.place(x=770, y=30, height=100, width=30)
btn27 = tk1.Button(root, text='', bg='white', command=lambda:btn1(46))
btn27.place(x=800, y=30, height=100, width=30)
btn28 = tk1.Button(root, text='', bg='white', command=lambda:btn1(48))
btn28.place(x=830, y=30, height=100, width=30)

btn29 = tk1.Button(root, text='', bg='white', command=lambda:btn1(49))
btn29.place(x=860, y=30, height=100, width=30)

btn51 = tk1.Button(root, bg='black', command=lambda:btn1(2))
btn51.place(x=40, y=30, height=60, width=20)
btn52 = tk1.Button(root, bg='black', command=lambda:btn1(4))
btn52.place(x=70, y=30, height=60, width=20)
btn53 = tk1.Button(root, bg='black', command=lambda:btn1(7))
btn53.place(x=130, y=30, height=60, width=20)
btn54 = tk1.Button(root, bg='black', command=lambda:btn1(9))
btn54.place(x=160, y=30, height=60, width=20)
btn55 = tk1.Button(root, bg='black', command=lambda:btn1(11))
btn55.place(x=190, y=30, height=60, width=20)

btn56 = tk1.Button(root, bg='black', command=lambda:btn1(14))
btn56.place(x=250, y=30, height=60, width=20)
btn57 = tk1.Button(root, bg='black', command=lambda:btn1(16))
btn57.place(x=280, y=30, height=60, width=20)
btn58 = tk1.Button(root, bg='black', command=lambda:btn1(19))
btn58.place(x=340, y=30, height=60, width=20)
btn59 = tk1.Button(root, bg='black', command=lambda:btn1(21))
btn59.place(x=370, y=30, height=60, width=20)
btn60 = tk1.Button(root, bg='black', command=lambda:btn1(23))
btn60.place(x=400, y=30, height=60, width=20)

btn61 = tk1.Button(root, bg='black', command=lambda:btn1(26))
btn61.place(x=460, y=30, height=60, width=20)
btn62 = tk1.Button(root, bg='black', command=lambda:btn1(28))
btn62.place(x=490, y=30, height=60, width=20)
btn63 = tk1.Button(root, bg='black', command=lambda:btn1(31))
btn63.place(x=550, y=30, height=60, width=20)
btn64 = tk1.Button(root, bg='black', command=lambda:btn1(33))
btn64.place(x=580, y=30, height=60, width=20)
btn65 = tk1.Button(root, bg='black', command=lambda:btn1(35))
btn65.place(x=610, y=30, height=60, width=20)

btn66 = tk1.Button(root, bg='black', command=lambda:btn1(38))
btn66.place(x=670, y=30, height=60, width=20)
btn67 = tk1.Button(root, bg='black', command=lambda:btn1(40))
btn67.place(x=700, y=30, height=60, width=20)
btn68 = tk1.Button(root, bg='black', command=lambda:btn1(43))
btn68.place(x=760, y=30, height=60, width=20)
btn69 = tk1.Button(root, bg='black', command=lambda:btn1(45))
btn69.place(x=790, y=30, height=60, width=20)
btn70 = tk1.Button(root, bg='black', command=lambda:btn1(47))
btn70.place(x=820, y=30, height=60, width=20)

label1 = tk1.Label(text='instrument :')
label1.place(x=310, y=5, height=20, width=80) 
label1["bg"] = "#202020" 
label1["foreground"] = "#FFFFFF" 
textbox1 = tk1.Entry( master=root ) 
textbox1.place( x=390, y=5, height=20, width=30) 
btn_inst1 = tk1.Button(root, text='-', command=lambda:inst1(-1))
btn_inst1.place(x=420, y=5, height=20, width=20)
btn_inst1["bg"] = "#444444" 
btn_inst1["foreground"] = "#FFFFFF" 
btn_inst2 = tk1.Button(root, text='+', command=lambda:inst1(1))
btn_inst2.place(x=440, y=5, height=20, width=20)
btn_inst2["bg"] = "#444444" 
btn_inst2["foreground"] = "#FFFFFF" 

label2 = tk1.Label(text='volume :')
label2["bg"] = "#202020" 
label2["foreground"] = "#FFFFFF" 
label2.place(x=510, y=5, height=20, width=70) 
textbox2 = tk1.Entry( master=root ) 
textbox2.place( x=580, y=5, height=20, width=30) 
btn_inst3 = tk1.Button(root, text='-', command=lambda:volu1(-1))
btn_inst3.place(x=610, y=5, height=20, width=20)
btn_inst3["bg"] = "#444444" 
btn_inst3["foreground"] = "#FFFFFF" 
btn_inst4 = tk1.Button(root, text='+', command=lambda:volu1(1))
btn_inst4.place(x=630, y=5, height=20, width=20)
btn_inst4["bg"] = "#444444" 
btn_inst4["foreground"] = "#FFFFFF" 

ins1 = 1 
vol1 = 127 
oct1 = 0 
chn1 = 0 
n1 = 60
pygame.midi.init()
out1 = pygame.midi.Output(0)
out1.set_instrument(ins1, chn1)
textbox1.insert(0, ins1) 
textbox2.insert(0, vol1) 

root.mainloop()

out1.close()
pygame.midi.quit()
