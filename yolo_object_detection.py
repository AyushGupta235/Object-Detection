import cv2
import numpy as np
import argparse
import time
import numba
from numba import cuda

#ap = argparse.ArgumentParser()
#args = ap.parse_args()

# Loading the Model
def load_yolo():

    # Loading the weights and configuration files
    net = cv2.dnn.readNet("C:\\Users\\ayush\\Jupyter\\Object Detection Using YOLOv3\\yolov3.weights", "C:\\Users\\ayush\\Jupyter\\Object Detection Using YOLOv3\\yolov3.cfg")

    # Creating a list of class names
    classes = []
    with open("C:\\Users\\ayush\\Jupyter\\Object Detection Using YOLOv3\\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Getting the names of the layers of YOLOv3
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size = (len(classes), 3))   

    return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels


# 1. Preprocess the image by setting the resolution of the image to a fixed size of 320 px by 320 px and scaling the RGB values from (0, 255) to (0, 1)
# 2. Pass the image (blob) forward through the network and receive the information in a (n x 85) matrix (n is the number of grid cells)
def detect_objects(path, net, output_layers):
	blob = cv2.dnn.blobFromImage(path, scalefactor = 0.00392, size = (320, 320), mean = (0, 0, 0), swapRB = True, crop = False)
	net.setInput(blob)
	outputs = net.forward(output_layers)
	return blob, outputs


# Extract the box dimensions from the output of the network
def get_box_dimensions(outputs, height, width):
	boxes = []
	confidence = []
	class_ids = []

	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				c_x = int(detect[0] * width)
				c_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(c_x - (w / 2))
				y = int(c_y - (h / 2))
				boxes.append([x, y, w, h])
				confidence.append(conf)
				class_ids.append(class_id)

	return boxes, confidence, class_ids


# Draw labels and label name on the selected boxes after Non-Maximum Suppression
def draw_labels(boxes, confidence, colors, class_ids, classes, path):
	indices = cv2.dnn.NMSBoxes(boxes, confidence, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	
	for i in range(len(boxes)):
		if i in indices:
			x, y, w, h = boxes[i]
			label = classes[class_ids[i]]
			conf = confidence[i]
			color = colors[i]
			cv2.rectangle(path, (x, y), (x + w, y + h), color, 2)
			cv2.putText(path, label, (x, y - 10), font, 1, color, 1)

	cv2.imshow("Video", path)

def image_detect(path):
	model, classes, colors, output_layers = load_yolo()
	video, height, width, channels = load_image(path)
	blobs, outputs = detect_objects(video, model, output_layers)
	boxes, confidence, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confidence, colors, class_ids, classes, video)

	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break
		cv2.destroyAllWindows()



# Detecting objects using webcam live
def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(0)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


# Detecting objects in a video
def start_video(video_path = "C:\\Users\\ayush\\Videos\\Captures\\wood 3 killing platinum 3.mp4"):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


if __name__ == '__main__':
	video_path = "D:\\CS Material\\ML\\Projects\\Object Detector\\test_video.mp4"
	start_video(video_path)

	cv2.destroyAllWindows()

#video_detect("D:\\CS Material\\ML\\Projects\\Object Detector\\test_video.mp4")

#cv2.imshow("Image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()