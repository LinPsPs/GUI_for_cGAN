from glob import glob
import numpy as np
from matplotlib import pylab as plt
import cv2
import tensorflow as tf
print(tf.__version__)
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import time
import os
from keras.models import load_model
import warnings
from warnings import simplefilter

from tkinter import *
from tkinter import ttk, colorchooser, filedialog
import enum
from PIL import Image
from PIL import ImageTk
import cv2

class Tool(enum.Enum):
    PENCIL = 1,
    RECT = 2,
    CIRCLE = 3

class Mode(enum.Enum):
    DRAWER = 1,
    TRAIN = 2


class cGanUI:
    def __init__(self, master):
        self.master = master
        self.drawerControls = Frame(self.master, padx=5, pady=5)
        self.trainControls = Frame(self.master, padx=5, pady=5)
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.tool = Tool.PENCIL
        self.mode = None
        self.penwidth = 5
        self.initWidgets()        

    def paint(self, e):
        if self.old_x and self.old_y and self.tool == Tool.PENCIL:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg,
                capstyle=ROUND, smooth=True)
        if self.old_x and self.old_y and (self.tool != Tool.PENCIL):
            return
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # reseting or cleaning the canvas
        if self.tool == Tool.RECT:
            self.c.create_rectangle(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, outline=self.color_fg) #, fill=self.color_fg)
        elif self.tool == Tool.CIRCLE:
            self.c.create_oval(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, outline=self.color_fg) #, fill=self.color_fg)
        self.old_x = None
        self.old_y = None
    
    def resetCanvas(self):
        self.drawerControls.pack(side=LEFT)
        self.c.pack(fill=BOTH, expand=True)
        if(panelA != None):
            panelA.pack_forget()
        if(panelB != None):
            panelB.pack_forget()
        self.trainControls.pack_forget()

    def changeW(self, e):  # change Width of pen through slider
        self.penwidth = e

    def updateToolPEN(self):
        self.tool = Tool.PENCIL
    
    def updateToolRECT(self):
        self.tool = Tool.RECT

    def updateToolCIRCLE(self):
        self.tool = Tool.CIRCLE

    def clear(self):
        self.c.delete(ALL)

    def change_fg(self):  # changing the pen color
        self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  # changing the background color canvas
        self.color_bg = colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def save_as_png(self):
        # save postscipt image 
        self.c.postscript(file='output.eps') 
        # use PIL to convert to PNG 
        img = Image.open('output.eps') 
        img.save('output.png', 'png') 

    def initWidgets(self):
        self.drawDrawerWidgets()
        # self.drawTrainWidgets()
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Save Cavas', command=self.save_as_png)
        filemenu.add_command(label='Load', command=self.select_image)
        filemenu.add_command(label='Reset', command=self.resetCanvas)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors', menu=colormenu)
        colormenu.add_command(label='Brush Color', command=self.change_fg)
        colormenu.add_command(label='Background Color', command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas', command=self.clear)
        optionmenu.add_command(label='Exit', command=self.master.destroy)
        # modemenu = Menu(menu)
        # menu.add_cascade(label='Mode', menu=modemenu)
        # modemenu.add_command(label='Drawer', command=self.drawDrawerWidgets)
        # modemenu.add_command(label='Train', command=self.drawTrainWidgets)

    def drawDrawerWidgets(self):
        Label(self.drawerControls, text='Pen Width', font=('arial 12')).grid(row=0, column=0)
        self.slider = ttk.Scale(self.drawerControls, from_=5, to=100, command=self.changeW, orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=1, column=0, ipadx=70, pady=20)
        Button(self.drawerControls, text="Pencil", command=self.updateToolPEN).grid(row=3, column=0, pady=20)
        Button(self.drawerControls, text="Rect", command=self.updateToolRECT).grid(row=4, column=0, ipadx = 5, pady=20)
        Button(self.drawerControls, text="Circle", command=self.updateToolCIRCLE).grid(row=5, column=0, pady=20)
        self.drawerControls.pack(side=LEFT)
        self.c = Canvas(self.master, width=800, height=800, bg=self.color_bg, )
        self.c.pack(fill=BOTH, expand=True)
        self.c.bind('<B1-Motion>', self.paint)  # drwaing the line
        self.c.bind('<ButtonRelease-1>', self.reset)

    def generate(self, path):
        # open a file chooser dialog and allow the user to select an input
        img_A = []
        img1 = cv2.imread(path)
        img1 = img1[..., ::-1]
        img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
        img_A.append(img1)
        img_A = np.array(img_A) / 127.5 - 1

        generator = load_model('Ver_02.h5')

        fake_A = generator.predict(img_A)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * fake_A + 0.5

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(gen_imgs[0])
        plt.axis('off')
        plt.savefig("test.png")


    def select_image(self):
        self.drawerControls.pack_forget()
        self.c.pack_forget()
        self.trainControls.pack(side=RIGHT)
        global panelA, panelB

        # open a file chooser dialog and allow the user to select an input
        path = filedialog.askopenfilename()
        # ensure a file path was selected
        if len(path) > 0:
            image = cv2.imread(path)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            out = self.generate(path)

            out = cv2.imread('test.png')
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out = Image.fromarray(out)
            out = ImageTk.PhotoImage(out)

            # if the panels are None, initialize them
            if panelA is None or panelB is None:
                # the first panel will store our original image
                panelA = Label(self.trainControls, image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
                
                panelB = Label(self.trainControls, image=out)
                panelB.image = out
                panelB.pack(side="right", padx=10, pady=10)
            # otherwise, update the image panels
            else:
                # update the pannels
                panelA.configure(image=image)
                panelB.configure(image=out)
                panelA.image = image
                panelB.image = out
                panelA.pack(side="left", padx=10, pady=10)
                panelB.pack(side="right", padx=10, pady=10)


if __name__ == '__main__':
    root = Tk()
    panelA = None
    panelB = None
    cGanUI(root)
    root.title('GAN')
    root.mainloop()
