import tkinter as tk
from PIL import ImageTk, Image
import threading
import logging
from vqganclip import generate_image

# important variables
window = tk.Tk()
description = tk.StringVar()
global img

class LockedImage:
    def __init__(self, initial_img):
        self.img = initial_img
        self._lock = threading.Lock()

    def locked_update(self, value):
        with self._lock:
            self.img = value
    def locked_get(self):
        img = None
        with self._lock:
            img = self.img.copy()
        return img

synced_image = LockedImage(Image.open("ny.jpg"))

def vqganclip_thread(prompts, locked_image):
    generate_image(prompts, locked_image)

def update_image(locked_image):
        print("Updating gui")
        image = locked_image.locked_get()
        resized = image.resize((450, 450))
        window.generated_image = ImageTk.PhotoImage(resized)
        img_label.configure(image=window.generated_image)
        window.after(4000, update_image, locked_image) 


# stores the user's image description
def submit():
    text = description.get()
    if text != "":
        print("Start thread")
        x = threading.Thread(target=vqganclip_thread, args=([text], synced_image), daemon=True)
        x.start()


# set's current displayed image to generated image

def set_image():
    window.generated_image = ImageTk.PhotoImage(Image.open("ny.jpg"))
    img_label.configure(image=window.generated_image)


# clears current text in the entry box
def clear_text():
    description.set("")


# makes window
window.title("Text to Image Generator")
window.geometry('725x625')

# makes image placeholder
image = Image.open("ny.jpg")
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

window.after(4000, update_image, synced_image)
window.mainloop()