import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

# Import the custom model
from ultrafastLaneDetector.model import parsingNet

# Define the lane colors to be used for visualization
lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

# Define the row anchors for the TUSIMPLE and CULANE datasets
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
                        116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                        168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                        220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                        272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

# Define an enumeration for the two model types
class ModelType(Enum):
    TUSIMPLE = 0
    CULANE = 1

# Define a class for the model configuration
class ModelConfig():

    def __init__(self, model_type):
        # Call the appropriate init function based on the model type
        if model_type == ModelType.TUSIMPLE:
            self.init_tusimple_config()
        else:
            self.init_culane_config()

    def init_tusimple_config(self):
        # Set the image width and height
        self.img_w = 1280
        self.img_h = 720
        # Set the row anchor and griding number for the TUSIMPLE dataset
        self.row_anchor = tusimple_row_anchor
        self.griding_num = 100
        # Set the number of classes per lane
        self.cls_num_per_lane = 56

    def init_culane_config(self):
        # Set the image width and height
        self.img_w = 1640
        self.img_h = 590
        # Set the row anchor and griding number for the CULANE dataset
        self.row_anchor = culane_row_anchor
        self.griding_num = 200
        # Set the number of classes per lane
        self.cls_num_per_lane = 18
class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model'] # Apple GPU
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB format
		img_pil = Image.fromarray(img)  # Create a PIL Image object from the numpy array representation of the image
		input_img = self.img_transform(img_pil)  # Apply transformations to the image using the img_transform method
		input_tensor = input_img[None, ...]  # Add an extra dimension to the tensor using None and ellipsis

		if self.use_gpu:
			if not torch.backends.mps.is_built():  # Check if the MPS backend is built
				input_tensor = input_tensor.cuda()  # Move the tensor to GPU memory if available

		return input_tensor


	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)

		return output

	@staticmethod
# This function processes the output of a lane detection model
	def process_output(output, cfg):

		# Parse the output of the model and reverse the order of the columns
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]

		# Apply softmax activation function to the output and calculate the location of each lane point
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)

		# Convert the output to an array of lane points
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc

		# Calculate the sampling points for the columns and the width of each column
		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		# Initialize empty lists for the lane points and detection status
		lanes_points = []
		lanes_detected = []

		# Iterate through each lane
		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []

			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:
				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						# Calculate the x and y coordinates of the lane point and append to the list
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			# Append the lane points to the list of lane points
			lanes_points.append(lane_points)

		# Convert the lane points and detection status to arrays and return them
		return np.array(lanes_points), np.array(lanes_detected)


	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		# Resize the input image for visualization
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# Draw lane points in the image
		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					# Draw a circle at each detected lane point
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
			for lane_num,lane_points in enumerate(lanes_points):
				if len(lane_points) > 1:
					for i in range(len(lane_points)-1):
						# Draw a line connecting each lane point
						start_point = (lane_points[i][0], lane_points[i][1])
						end_point = (lane_points[i+1][0], lane_points[i+1][1])
						cv2.line(visualization_img, start_point, end_point, lane_colors[lane_num], thickness=2)

		return visualization_img



	







