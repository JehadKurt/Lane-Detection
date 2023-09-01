import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

class LaneDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Lane Detection")

        # Create input image label
        self.input_label = tk.Label(master, text="Input Directory:")
        self.input_label.grid(row=0, column=0)

        # Create input image entry box
        self.input_entry = tk.Entry(master)
        self.input_entry.grid(row=0, column=1)

        # Create select input directory button
        self.select_input_button = tk.Button(master, text="Select Directory", command=self.select_input_directory)
        self.select_input_button.grid(row=0, column=2)

        # Create output image label
        self.output_label = tk.Label(master, text="Output Directory:")
        self.output_label.grid(row=1, column=0)

        # Create output image entry box
        self.output_entry = tk.Entry(master)
        self.output_entry.grid(row=1, column=1)

        # Create select output directory button
        self.select_output_button = tk.Button(master, text="Select Directory", command=self.select_output_directory)
        self.select_output_button.grid(row=1, column=2)

        # Create detect lanes button
        self.detect_button = tk.Button(master, text="Detect Lanes", command=self.detect_lanes)
        self.detect_button.grid(row=2, column=0)

        # Create quit button
        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=1)

    def select_input_directory(self):
        # Open file dialog to select directory
        directory = filedialog.askdirectory()

        # Set selected directory to input entry box
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, directory)

    def select_output_directory(self):
        # Open file dialog to select directory
        directory = filedialog.askdirectory()

        # Set selected directory to output entry box
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, directory)

    def detect_lanes(self):
        # Get input and output directories from entry boxes
        input_directory = self.input_entry.get()
        output_directory = self.output_entry.get()

        # Initialize lane detection model
        model_path = "models/tusimple_18.pth"
        model_type = ModelType.TUSIMPLE
        use_gpu = False
        lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

        # Get a list of image files in the input directory
        input_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.bmp')]

        for input_file in input_files:
            # Construct output file path by replacing input directory with output directory
            output_file = input_file.replace(input_directory, output_directory)

            # Read input image
            img = cv2.imread(input_file, cv2.IMREAD_COLOR)

            # Detect lanes in the image
            output_img = lane_detector.detect_lanes(img)

            # Save output image
            cv2.imwrite(output_file, output_img)
        return None
# Create a Tkinter window and start the event loop
root = tk.Tk()
gui = LaneDetectionGUI(root)
root.mainloop()
