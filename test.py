import cv2, sys
from bam import *
from training import *
from bwmorph_thin import *

TRAINING_DIR = 'letters/A/'

width = 32
height = 32
num_rows = 32*32
num_cols = num_rows*3 - 1


def zero_one_to_image (arr):
	image_vector = np.zeros((height, width), np.float32)
	for row in range(image_vector.shape[0]):
		for col in range(image_vector.shape[1]):
			if arr[row,col] == 0:
				image_vector[row, col] = 255
			else:
				image_vector[row, col] = 0
	cv2.imshow('zero one', image_vector)
	return image_vector
	
def thin (img):
	image = []
	for row in range(img.shape[0]):
		v = []
		for col in range(img.shape[1]):
			val = img[row,col]
			if val==0:
				v.append(1)
			else:
				v.append(0)
				
		image.append(v)
		
	# skel = np.array(image).astype(np.uint8)
	# print skel
	x = bwmorph_thin(image)
	new_image = zero_one_to_image(x)
	return new_image
	# cv2.imshow('zero one', x)
	
def get_bipolar_vector (img):
	cv2.imshow('orig', img)
	img = cv2.resize(img, (width, height))
	ret2, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	cv2.imshow('bw', th2)
	
	thinned_image = thin(th2)
	
	bipolar_vector = []
	for row in range(th2.shape[0]):
		for col in range(th2.shape[1]):
			val = th2[row,col]
			if val == 0:
				bipolar_vector.append(-1)
			else:
				bipolar_vector.append(1)
				
	# cv2.imshow('bnw',th2)
	return bipolar_vector
	
training_set = load_images_from_folder (TRAINING_DIR)

for image in training_set:
	bp = get_bipolar_vector(image)
	
cv2.waitKey(0)
cv2.destroyAllWindows()