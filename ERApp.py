import customtkinter
import tkinter as tk
from PIL import ImageTk, Image
from engine import *
import numpy as np
import cv2


class_dict = {
    "no face": 0,
    "happy": 1,
    "sad": 2,
    "natural": 3,
    "surprised": 4,
    "angry": 5,
}

reverse_dict = {v: k for k, v in class_dict.items()}

face_tracker = build_model()
model = FaceTracker(face_tracker)
model.built = True
model.load_weights("models/ER_model.h5")


class VidFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.label = tk.Label(self, width=master.winfo_width(), height=master.winfo_height(), bg="black")
        #self.label.grid(row=0, column=0, sticky="nsew")
        self.label.pack()
        


class ERApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Emotional Recognition")
        self.geometry("640x480")
        self.resizable(False, False)

        self.vid_frame = VidFrame(self, width=self.winfo_width(), height=self.winfo_height())
        self.vid_frame.pack()


def open_cam(vid):
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    # frame = frame[50:500, 50:500, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    frame = cv2.resize(frame, (width, height))

    # *Model prediction
    Snapshot = frame[50:500, 50:500, :]
    Snapshot = np.expand_dims(cv2.resize(Snapshot, (120, 120)) / 255.0, axis=0)
    pred = model.predict(Snapshot)
    coords = pred[1][0]
    if pred[0].argmax() != 0:
        cv2.rectangle(
            frame,
            pt1=(
                int(coords[0] * frame.shape[1]) - 50,
                int(coords[1] * frame.shape[0]) + 50,
            ),
            pt2=(
                int(coords[2] * frame.shape[1] - 50),
                int(coords[3] * frame.shape[0]) + 50,
            ),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            reverse_dict[pred[0].argmax(axis=1)[0]],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # * Convert to ImageTk format
    image = Image.fromarray(frame)
    image = ImageTk.PhotoImage(image)

    # * Display
    app.vid_frame.label.configure(image=image)
    app.vid_frame.label.image = image

    # * Repeat
    app.after(10, open_cam, vid)


if __name__ == "__main__":
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height = 640, 480
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    app = ERApp()
    app.bind("<Escape>", lambda e: app.destroy())
    open_cam(vid)
    app.mainloop()
