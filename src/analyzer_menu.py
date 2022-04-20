from tkinter import filedialog
import os
from tkinter import Tk

Tk().withdraw()

dir_path = "/Users/markusjonek/Documents/golf-analys"
init_path = dir_path + "/golf_videos" 
exe_path = dir_path + "/build_golf/golf_analyzer"

filepath = filedialog.askopenfilename(initialdir=init_path,
                                      title="Golf analyzer",
                                      filetypes=(("", "*.MOV"),("", "*.mp4"),
                                      ("all files", "*.*")))


os.system(exe_path + " " + filepath) 
