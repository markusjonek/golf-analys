from tkinter import filedialog
import os
from tkinter import Tk

Tk().withdraw()
filepath = filedialog.askopenfilename(initialdir=os.getcwd(),
                                      title="Golf analyzer",
                                      filetypes=(("", "*.MOV"),("", "*.mp4"),
                                      ("all files", "*.*")))


os.system("/Users/markusjonek/Documents/golf-analys/cmake-build-golf_analyzer/obj_det " + filepath) 
