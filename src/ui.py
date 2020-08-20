# author Haolin

from tkinter import Tk as tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
import enum

# enum tools
class Tool(enum.Enum):
    LINE = 1
    OVAL = 2
    RECT = 3
    

root = tk()

root.title("cGAN")

frame = tk.Frame(root)
frame.pack(fill=BOTH, expand=True)

frame.grid_columnconfigure(1, weight=1)
frame.grid_rowconfigure(3, weight=1)

canvas = tk.Canvas(root, bg="white", width=800, height=600)
canvas.pack()
canvas.grid(row=1, column=0, columnspan=2, rowspan=4, padx=5, sticky=E+W+S+N)


coords = {"x":0,"y":0,"x2":0,"y2":0}
# keep a reference to all lines by keeping them in a list 
lines = []

def click(e):
    # define start point for line
    coords["x"] = e.x
    coords["y"] = e.y

    # create a line on this point and store it in the list
    lines.append(canvas.create_line(coords["x"],coords["y"],coords["x"],coords["y"]))

def drag(e):
    # update the coordinates from the event
    coords["x2"] = e.x
    coords["y2"] = e.y

    # Change the coordinates of the last created line to the new coordinates
    canvas.coords(lines[-1], coords["x"],coords["y"],coords["x2"],coords["y2"])

canvas.bind("<ButtonPress-1>", click)
canvas.bind("<B1-Motion>", drag) 

root.mainloop()