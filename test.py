import cv2, sys
from bam import *
from training import *
from bwmorph_thin import *

TRAINING_DIR = 'letters/A/'

width = 32
height = 32
num_rows = width*height
num_cols = num_rows*3 - 1

def get_features (img):
	# Gets height-width ratio and upper ratio
	top_bound_row = left_bound_row = height
	top_bound_col = left_bound_col = width
	bottom_bound_row = right_bound_row = bottom_bound_col = right_bound_col = 0
	
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			val = img[row, col]
			if val==0:
				if col <= left_bound_col:
					left_bound_col = col
					left_bound_row = row
				if col >= right_bound_col:
					right_bound_col = col
					right_bound_row = row
				if row <= top_bound_row:
					top_bound_row = row
					top_bound_col = col
				if row >= bottom_bound_row:
					bottom_bound_row = row
					bottom_bound_col = col
	
	character_height = bottom_bound_row - top_bound_row
	character_width = right_bound_col - left_bound_col
	
	center_row = ( character_height/2 ) + top_bound_row
	print center_row
	
	total_pixels = 0
	upper_pixels = 0
	
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			val = img[row, col]
			if val==0:
				if row<=center_row:
					upper_pixels+=1
				total_pixels+=1
				
	
	cv2.imshow('bounds', img)
	
	height_width_ratio = character_height*1.0 / character_width
	upper_ratio = upper_pixels*1.0 / total_pixels
	
	print character_height, character_width, height_width_ratio
	
	print upper_pixels, total_pixels, upper_ratio
	return (height_width_ratio, upper_ratio)
	

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
	
	cv2.imshow('wb', th2)
	
	thinned_image = thin(th2)
	
	get_features(thinned_image)
	
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