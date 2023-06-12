import customtkinter
from PIL import Image
from TryModule import *
import pyautogui, time


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("1920x1080")



def SScreenshot():
    time.sleep(5)
    print("ScreenShot taked..")
    screenshot = pyautogui.screenshot()
    # Save the screenshot to a file
    screenshot.save('screenshot.png')
    Learing()
    time.sleep(10)
    

# make UI Frame linked to root
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

#Text
label = customtkinter.CTkLabel(master=frame, text="A.I Project", font=("Roboto",50))
label.pack(pady=12, padx=10)

#image load
imageDir = ('img/OLD/Part2/img/image_T0.png')

button = customtkinter.CTkButton(master=frame, text ="TakeScreenShot",font=("Roboto",25), command=SScreenshot)
button.pack(pady=12, padx=10)

immage = customtkinter.CTkImage(dark_image=Image.open(imageDir),size=(1920,1080))
label2 = customtkinter.CTkLabel(master=frame, image=immage)
label2.pack(pady=12, padx=10)


#openWindow
root.mainloop()

