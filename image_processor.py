from math import sqrt, sin, cos, atan2, floor
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json


def dist(a, b):
    return sqrt(
        (b[0] - a[0])**2 +
        (b[1] - a[1])**2
    )

def fix_rotation(image, left_eye, right_eye):
    rot = -atan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    )

    x, y = left_eye
    c = cos(rot)/1
    s = sin(rot)/1
    return image.transform(
        image.size,
        Image.AFFINE,
        (
            c, s, x - x*c - y*s,
            -s, c, y - x*-s - y*c
        ),
        resample=Image.BICUBIC
    )

def fix_alignment(image, left_eye, right_eye):
    scale = dist(left_eye, right_eye) / 26
    x = left_eye[0] - scale * 19
    y = left_eye[1] - scale * 19
    size = 64 * scale

    return image.crop((
        int(x),
        int(y),
        int(x + size),
        int(y + size)
    ))

def fix_size(image):
    return image.resize((64, 64), Image.ANTIALIAS)

def format_face(path, left_eye, right_eye, dest_path=None):
    # use source path as destination path if no dest provided
    if not dest_path:
        dest_path = path

    image = Image.open(path)
    image = fix_rotation(image, left_eye, right_eye)
    image = fix_alignment(image, left_eye, right_eye)
    image = fix_size(image)
    image.save(dest_path)

class EyeLocator:
    def __init__(self, image_data, source_dir, dest_dir, plt, index=0):
        self.images = []
        for subject in image_data:
            for image_name in subject:
                self.images.append(image_name)

        self.master = plt
        self.fig = self.master.figure()
        self.dest_dir = dest_dir
        self.source_dir = source_dir
        self.index = index
        self.enable_events()
        self.next_face()

    def enable_events(self):
        self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        # no data recorded for image
        if self.status == 0:
            self.left_eye = (int(event.xdata), int(event.ydata))
            self.status = 1

        # left eye data recorded for image
        elif self.status == 1:
            self.right_eye = (int(event.xdata), int(event.ydata))
            self.status = 0
            self.finished_gathering()

    def finished_gathering(self):
        # process, increment index, display new face
        self.process_data()
        self.index += 1
        if self.index == len(self.images):
            self.master.close()
        self.next_face()

    def process_data(self):
        try:
            dest_path = self.dest_dir + "\\" + self.images[self.index]
            source_path = self.source_dir + "\\" + self.images[self.index]
            format_face(
                source_path,
                self.left_eye,
                self.right_eye,
                dest_path
            )
        except FileNotFoundError:
            self.master.close()
            print("You must create the directory `{}`".format(
                self.dest_dir
            ))
            input()  # stop program from instantly closing
            exit()

    def reset(self):
        self.left_eye = None
        self.right_eye = None
        self.status = 0

    def next_face(self):
        # reset measurement variables
        self.reset()

        # load next face
        file_name = self.source_dir + "\\" + self.images[self.index]
        img = mpimg.imread(file_name)

        # draw face
        self.master.clf()
        self.master.imshow(img, cmap="gray")
        self.fig.canvas.draw()

    def start(self):
        # start matplotlib loop
        self.master.show()


if __name__ == "__main__":
    # load image data from file
    with open("image_data.json") as f:
        image_data = json.load(f)

    unprocessed_dir = "unprocessed_images"
    processed_dir = "processed_images"

    EyeLocator(image_data, unprocessed_dir, processed_dir, plt).start()
