# This block will set up the YOLOv4 algorithm class
# You need to run this block at least once

import numpy as np
import cv2
import time
import math
import random

COLORS = np.random.uniform(0, 255, size=500)


class ImgProc():
    def __init__(self):
        # read pre-trained model and config file
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

        # read class names from text file
        self.classes = None
        with open("coco.names", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes
        # self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    # function to get the output layer names
    # in the architecture
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in [net.getUnconnectedOutLayers()]]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect_objects(self, img):
        W = img.shape[1]
        H = img.shape[0]

        # create input blob
        sz = (416, 416)  # (224,224)
        normalization = 1.0 / 255.0
        blob = cv2.dnn.blobFromImage(img, normalization, sz, (0, 0, 0), True, crop=False)

        # set input blob for the network
        self.net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.get_output_layers(self.net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        centroids = []
        conf_threshold = 0.3
        nms_threshold = 0.1

        # For each detetion from each output layer get the confidence, class id, bounding box params and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    centroids.append((center_x, center_y))

        # Apply non-max suppression to prevent duplicate detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Go through the detections remaining after NMS and draw bounding boxes
        detections = []
        frame = img.copy()
        for i in indices:
            if type(i) is list:
                i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

            detections.append((self.classes[class_ids[i]], centroids[i], box))

        # print("Detected Objects: ", detections)
        return detections, frame


# Simplistic image projection to map
# Processed heat map prep
density = np.zeros([792, 777])


def image_projection_simple(xp, yp, camera_name="Yard 1 Facing Staging 1"):
    coords = []

    if camera_name == "Yard 1 Facing Staging 1":
        coords = coords + [min((xp / 1920) * 460, 455)]
        coords = coords + [min((yp / 1080) * 500, 460)]
    if camera_name == "Yard 3 Facing Yard 4":
        coords = coords + [min((xp / 1920) * 290, 476)]
        coords = coords + [min((yp / 1080) * 400 + 95, 431)]
    if camera_name == "Yard 3 Overlook":
        coords = coords + [min((xp / 1920) * 425 * 0.5 - (yp / 1080) * 704 * 0.9 + 300, 423)]
        coords = coords + [min((yp / 1080) * 704 * 0.5 + (xp / 1920) * 425 * 0.9 + 200, 702)]
    else:
        coords = coords + [(xp / 1920) * 748 + 50]
        coords = coords + [(yp / 1080) * 548 + 50]
    return coords


movement_tether = (0, 0)
cm_counter = 0


def calculate_movement(previous_pcoords, pcoords, pix2feet):
    global movement_tether
    global cm_counter
    cm_counter = cm_counter + 1
    SAME_ELEPHANT_DISTANCE = 50  # This is the max pixel distance between frames to consider the elephants in each frame the same

    if cm_counter % 10 == 0:
        distance = math.dist(pcoords, previous_pcoords)
        if distance < SAME_ELEPHANT_DISTANCE:
            distance = math.dist(movement_tether, pcoords)
            movement_tether = previous_pcoords
            return distance * pix2feet
        else:
            return 0
    else:
        return 0


# Live Map Code
import matplotlib.pyplot as plt

# This is the app code

import tkinter as tk
from tkinter import filedialog

cam_list = ["Yard 1 Facing Staging 1", "Staging 1", "Yard 3 Facing Yard 4", "Yard 3 Overlook"]
instructions_text = "Instructions: Upload your MP4 file, your map PNG file, and choose the corresponding camera."


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.title = tk.Label(self, text="Zoober Elephant Video Analyzer", font=("Arial", 26))
        self.credit = tk.Label(self,
                               text="Made by Zoober Team at UCSD The Basement. Email salitman@ucsd.edu for help/feedback.",
                               font=("Arial", 8))
        self.instructions = tk.Label(self, text=instructions_text, font=("Arial", 10))
        self.uploader = tk.Button(self, text="Upload an MP4", fg="black", command=self.ask_file)
        self.analyzer2 = tk.Button(self, text="Generate Heat Map", fg="black", command=self.plain_map)
        self.cam_location = tk.StringVar()
        self.cam_location.set("Staging 1")
        self.drop = tk.OptionMenu(self, self.cam_location, *cam_list)
        self.map_chooser = tk.Button(self, text="Upload the corresponding map", command=self.ask_map)
        self.title.pack()
        self.instructions.pack()
        self.credit.pack()
        self.uploader.pack()
        self.map_chooser.pack()
        self.drop.pack()
        self.analyzer2.pack()

    def ask_file(self):
        root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                   filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        if root.filename:
            try:
                self.confirmation.destroy()
            except:
                pass
            try:
                self.distance_walked.destroy()
                self.time_walked.destroy()
                self.time_stationary.destroy()
                self.image_saved.destroy()
            except:
                pass
            self.confirmation = tk.Label(self, text="Video: " + root.filename, font=("Arial", 16), fg="grey")
            self.confirmation.pack()

    def ask_map(self):
        root.mapfilename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                      filetypes=(("png files", "*.png"), ("all files", "*.*")))
        if root.mapfilename:
            try:
                self.mapconfirmation.destroy()
            except:
                pass
            self.mapconfirmation = tk.Label(self, text="Map: " + root.mapfilename, font=("Arial", 16), fg="grey")
            self.mapconfirmation.pack()

    def abort(self):
        self.stop_plainmap = 1

    def display_data(self, total_distance, total_time, total_stationary, FRAMES_PER_SECOND):
        # FRAMES_PER_SECOND = 10
        time_walked = int(total_time / FRAMES_PER_SECOND)
        time_stationary = int(total_stationary / FRAMES_PER_SECOND)
        self.distance_walked = tk.Label(self, text="Elephant(s) walked " + str(
            int(total_distance)) + "ft. This is a very rough estimate.", font=("Arial", 16))
        self.time_walked = tk.Label(self, text="Elephant(s) walked for " + str(time_stationary) + "s.",
                                    font=("Arial", 16))
        self.time_stationary = tk.Label(self, text="Elephant(s) were stationary for " + str(time_walked) + "s.",
                                        font=("Arial", 16))
        self.distance_walked.pack()
        self.time_walked.pack()
        self.time_stationary.pack()

    def plain_map(self):
        camera_name = self.cam_location.get()
        print("CAMERA LOCATION:", camera_name)
        self.stop_plainmap = 0
        cap = cv2.VideoCapture(root.filename)

        counter = 0
        N_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps / 5

        start_time = time.time()

        self.loading = tk.Label(text="PROCESSING... " + str(round(counter / N_frames, 3) * 100) + "%", fg="red",
                                font=("Arial", 15))
        self.loading.pack()
        self.timeelapsed = tk.Label(text="TIME ELAPSED: " + str(int(time.time() - start_time)) + "ms", fg="red",
                                    font=("Arial", 15))
        self.timeelapsed.pack()
        self.end_process = tk.Button(self, text="End Processing", fg="black", command=self.abort)
        self.end_process.pack()

        map_path = root.mapfilename

        try:
            self.image_saved.destroy()
        except:
            pass

        if camera_name == "Yard 1 Facing Staging 1":
            density = np.zeros([462, 457])
            pix2feet = 22 / 457
        if camera_name == "Yard 3 Facing Yard 4":
            density = np.zeros([433, 478])
            pix2feet = 100 / 433
        if camera_name == "Yard 3 Overlook":
            density = np.zeros([704, 425])
            pix2feet = 150 / 704
        if camera_name == "Staging 1":
            density = np.zeros([600, 800])
            pix2feet = 55 / 800

        pcoords = (0, 0)
        previous_pcoords = (0, 0)
        total_distance = 0
        total_time = 0
        total_stationary = 0
        # THIS IS THE MAIN LOOP FOR FRAME ANALYSIS
        while True:
            if self.stop_plainmap == 1:
                break

            self.loading.configure(text="PROCESSING... " + str(round(counter / N_frames, 3) * 100) + "%")
            self.timeelapsed.configure(text="TIME ELAPSED: " + str(int(time.time() - start_time)) + "s")
            root.update()
            counter = counter + 1

            ret, img = cap.read()
            imgProc = ImgProc()
            if img is not None:
                dettt, frameee = imgProc.detect_objects(img)
            else:
                break

            # Controls the number of frames to be skipped
            if counter % 5 != 0:
                continue

            # Use detections in this frame and add to map
            for i in dettt:
                if i[0] == 'elephant' or i[0] == 'person' or i[0] == 'dog' or i[0] == 'cow' or i[0] == 'sheep' or i[
                    0] == 'horse':
                    center = i[1]
                    x = center[0]
                    y = center[1]
                    pcoords = image_projection_simple(x, y, camera_name)

                    for j in [math.floor(pcoords[1]) - 1, math.floor(pcoords[1]), math.floor(pcoords[1]) + 1]:
                        for k in [math.floor(pcoords[0]) - 1, math.floor(pcoords[0]), math.floor(pcoords[0]) + 1]:
                            density[j, k] = density[j, k] + 1

            distance = calculate_movement(previous_pcoords, pcoords, pix2feet)
            total_distance = total_distance + distance
            if distance != 0:
                total_time = total_time + 1
            else:
                total_stationary = total_stationary + 1

            previous_pcoords = pcoords

        cap.release()

        try:
            self.loading.destroy()
            self.timeelapsed.destroy()
            self.end_process.destroy()
        except:
            pass

        plt.figure(num=2)
        plt.imshow(plt.imread(map_path))
        plt.imshow(density, cmap='hot', interpolation='nearest', alpha=0.7)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        image_filename = camera_name + "-" + str(random.randrange(1000)) + '.png'
        plt.savefig(image_filename, bbox_inches='tight')

        self.image_saved = tk.Label(self, text="HEATMAP SAVED AS " + image_filename, fg='red', font=("Arial", 15))
        self.image_saved.pack()

        self.display_data(total_distance, total_time, total_stationary, fps)


root = tk.Tk()
myapp = App(root)
myapp.master.title("Zoober Elephant Analysis")

myapp.master.minsize(1800, 900)
myapp.mainloop()