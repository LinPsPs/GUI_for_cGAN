from tkinter import *
from tkinter import ttk, colorchooser
import enum
from PIL import Image

class Tool(enum.Enum):
    PENCIL = 1,
    RECT = 2,
    CIRCLE = 3


class cGanUI:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.tool = Tool.PENCIL
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)  # drwaing the line
        self.c.bind('<ButtonRelease-1>', self.reset)

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

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        Label(self.controls, text='Pen Width', font=('arial 12')).grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=5, to=100, command=self.changeW, orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=1, column=0, ipadx=70, pady=20)
        Button(self.controls, text="Pencil", command=self.updateToolPEN).grid(row=3, column=0, pady=20)
        Button(self.controls, text="Rect", command=self.updateToolRECT).grid(row=4, column=0, ipadx = 5, pady=20)
        Button(self.controls, text="Circle", command=self.updateToolCIRCLE).grid(row=5, column=0, pady=20)
        self.controls.pack(side=LEFT)

        self.c = Canvas(self.master, width=800, height=600, bg=self.color_bg, )
        self.c.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Save Cavas', command=self.save_as_png)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors', menu=colormenu)
        colormenu.add_command(label='Brush Color', command=self.change_fg)
        colormenu.add_command(label='Background Color', command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas', command=self.clear)
        optionmenu.add_command(label='Exit', command=self.master.destroy)


if __name__ == '__main__':
    root = Tk()
    cGanUI(root)
    root.title('Application')
    root.mainloop()
