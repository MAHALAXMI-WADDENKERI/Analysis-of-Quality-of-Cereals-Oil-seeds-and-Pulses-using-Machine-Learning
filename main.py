from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import sqlite3
import os
root = Tk()
root.geometry('1366x768')
root.title("Seed")
Un = StringVar()
Pw = StringVar()
canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back - Copy.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(1,1, anchor=NW, image=photo)
def login():
    un = Un.get()
    pw = Pw.get()
    if un == "admin" and pw == "admin":
        root.destroy()
        os.system('python menu.py')


label_0 = Label(root, text="Login", bg='white', font=("bold", 20))
label_0.place(x=750, y=360)
label_4 = Label(root, text="Username", bg='white', font=("bold", 10))
label_4.place(x=750, y=420)
entry_5 = Entry(root, textvar=Un)
entry_5.place(x=850, y=420)
label_5 = Label(root, text="Password", bg='white',font=("bold", 10))
label_5.place(x=750, y=450)
entry_6 = Entry(root, textvar=Pw, show="*")
entry_6.place(x=850, y=450)
Button(root, text='Login', width=15, bg='green', fg='white', command=login).place(x=850, y=490)


root.mainloop()
