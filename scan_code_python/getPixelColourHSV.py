# import sys
# print(sys.executable)

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import colorsys

def open_image():
    global img, img_display, tk_img, canvas
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('RGB')
        img_display = ImageOps.contain(img, (600, 600))
        tk_img = ImageTk.PhotoImage(img_display)
        canvas.config(width=img_display.width, height=img_display.height)
        canvas.create_image(0, 0, anchor='nw', image=tk_img)

def get_pixel_color(event):
    x, y = event.x, event.y
    if img_display:
        pixel_rgb = img_display.getpixel((x, y))
        pixel_hsv = colorsys.rgb_to_hsv(pixel_rgb[0] / 255.0, pixel_rgb[1] / 255.0, pixel_rgb[2] / 255.0)
        hsv_str = f"HSV: ({pixel_hsv[0]*360:.2f}, {pixel_hsv[1]*100:.2f}%, {pixel_hsv[2]*100:.2f}%)"
        color_label.config(text=hsv_str)

# Set up the main window
root = tk.Tk()
root.title("Image Pixel HSV Selector")

# Create a canvas for the image
canvas = tk.Canvas(root, cursor="cross")
canvas.pack()

# Bind the mouse click event
canvas.bind("<Button-1>", get_pixel_color)

# Create a label to show the HSV value
color_label = tk.Label(root, text="HSV: ")
color_label.pack()

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Initialize global variables
img = None
img_display = None
tk_img = None

# Run the application
root.mainloop()
