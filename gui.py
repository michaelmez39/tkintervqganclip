import tkinter as tk
from PIL import ImageTk, Image

# important variables
window = tk.Tk()
description = tk.StringVar()
global img


# stores the user's image description
def submit():
    text = description.get()
    if text != "":
        print("Text entered: " + text)
        description.set("")
        set_image()


# set's current displayed image to generated image
def set_image():
    window.generated_image = ImageTk.PhotoImage(Image.open("images/img1.png"))
    img_label.configure(image=window.generated_image)


# clears current text in the entry box
def clear_text():
    description.set("")


# makes window
window.title("Text to Image Generator")
window.geometry('725x625')

# makes image placeholder
image = Image.open("images/img.png")
resize_image = image.resize((450, 450))
window.img = ImageTk.PhotoImage(resize_image)
img_label = tk.Label(window, image=window.img)
img_label.place(x=150, y=75)

# makes text entry box + label
tk.Label(window, text="Enter image description:", font='Aerial 12').place(x=5, y=12)
e1 = tk.Entry(window, textvariable=description, width=40, font='Aerial 12')
e1.place(x=180, y=14)

# makes buttons
tk.Button(window, text='Enter', command=submit, font='Aerial 10').place(x=550, y=10)
tk.Button(window, text='Clear', command=clear_text, font='Aerial 10').place(x=600, y=10)
tk.Button(window, text='Quit', command=window.quit, font='Aerial 12').place(x=5, y=575)

window.mainloop()