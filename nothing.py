"""
Evan Greene
2018-02-26

Some fun with OpenCV
"""

"""
A program to make a picture look sorta like a watercolor
"""
import numpy as np
import cv2

def equalizeBrightness(picture, brightness = 200):
	"""
	Takes a picture and makes the brightness equal for every pixel. The results
	are kinda trippy.
	"""
	# convert the picture to HSV color space.
	pictureHSV = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
	# set the value in HSV to a uniform 200. This makes the brightness of
	# every pixel identical.
	pictureHSV[:, :, 2] = brightness

	# convert back to BGR color space.
	newPicture = cv2.cvtColor(pictureHSV, cv2.COLOR_HSV2BGR)

	# cv2.imshow("brightness equalized", newPicture)
	# cv2.waitKey(0)
	print newPicture[0, 0]
	return newPicture

def equalizeSaturation(image):
	"""
	Takes as input an image and returns the image with the saturation uniform
	for every pixel.
	"""
	# convert to HSV
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# set the saturation value for every pixel to 200
	imageHSV[:, :, 1] = 200
	# convert back to BGR color
	return cv2.cvtColor(imageHSV,  cv2.COLOR_HSV2BGR)

def equalizeHue(image, hue = 100):
	"""
	Takes an image and returns the image with the value uniform for every pixel.
	"""
	# convert to HSV
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# set the hue for every pixel to the same number.
	imageHSV[:, :, 0] = hue
	# convert back to BGR and return.
	return cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)

def increaseValue(image, factor = 1.5):
	"""
	Brightens an image by a factor given in the second argument.
	"""
	# convert to HSV color space.
	# also convert to float32
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
	# multiply the value by a factor given in the argument.
	# print np.max(imageHSV[:, :, 2])
	imageHSV[:, :, 2] *= factor
	# print np.max(imageHSV[:, :, 2])
	imageHSV[:, :, 2] = np.clip(imageHSV[:, :, 2], 0, 255)
	# convert back to HSV color space.
	imageHSV = imageHSV.astype("uint8")


	image = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
	return image

def increaseSaturation(image, factor = 1.2):
	"""
	increases the saturation of an image by a factor given in the second
	argument
	"""
	# convert to HSV
	# convert to HSV color space.
	# also convert to float32
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
	# multiply the value by a factor given in the argument.
	# print np.max(imageHSV[:, :, 2])
	imageHSV[:, :, 1] *= factor
	# `print` np.max(imageHSV[:, :, 1])
	imageHSV[:, :, 1] = np.clip(imageHSV[:, :, 1], 0, 255)
	# convert back to HSV color space.
	imageHSV = imageHSV.astype("uint8")


	image = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
	return image

def playWithHue(image):
	"""
	does some weird things with the hue value in the HSV space.
	"""
	# first convert to HSV color space
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# code to quantize the hue values.
	# identify the centers of the hue.
	hueCenters = np.array([0, 28, 105, 180], dtype = 'int32')
	imageHSV[:, :, 0] = cluster(imageHSV[:, :, 0], hueCenters)

	# imageHSV[:, :, 0] = (imageHSV[:, :, 0] - 15) % 180

	# print imageHSV[20, 20]
	# imageHSV = makeKmeansPicture(imageHSV)

	return cv2.cvtColor(imageHSV.astype("uint8"), cv2.COLOR_HSV2BGR)

def playWithBrightness(image):
	"""
	does some weird things with the brightness value in HSV space.
	"""
	# convert to HSV
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# map the value to a minimum brighness of 64
	imageHSV[:, :, 2] = (((imageHSV[:, :, 2] / 4 ) * 3) + 64) % 255

	return cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)

def cluster(data, labels):
	"""
	returns a copy of data with each point moved to the closest value in labels
	"""
	# turn the data into a NumPy array if you haven't already
	data = np.array(data, dtype = "int32")
	# flatten the data so it behaves nicely.
	dataShape = data.shape
	data = data.flatten()

	# create a matrix that's len(labels) rows  x len(data) columns
	labelMatrix = np.empty((len(labels), len(data)))

	# find the distance from the data points to each label
	for i in range(len(labels)):
		labelMatrix[i, :] = abs(labels[i] - data)

	# convert the labels to a numpy array for this function to work.
	# then take the value of the numpy array the index where the minimum
	# value of that column of the labelMatrix.
	output = np.array(labels)[np.argmin(labelMatrix, axis = 0)]

	# finally, reshape it so that it looks like the input data did.
	return output.reshape(dataShape)

def makeKmeansPicture(picture):
	"""
	Takes a picture and uses k-means clustering to group the colors of the
	picture, then shows the resulting picture
	"""
	# apply a Gaussian Blur to the image
	# also convert to float32 format and reshape for clustering
	# data = cv2.GaussianBlur(picture, ksize = (5, 5), sigmaX = 2.0, sigmaY = 2.0)
	data = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
	data = data.reshape((-1, 3)).astype(np.float32)

	# specify the number of colors
	numColors = 10
	# the termination criteria for the kmeans clustering
	# the criteria here are a maximum of 10 iterations.
	criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# flags for the clustering
	flags = (cv2.KMEANS_PP_CENTERS)

	# do the actual clustering.
	_, labels, centers = cv2.kmeans(data, numColors, None, attempts = 5,
		criteria = criteria, flags = flags)

	# convert the centers to type uint8
	centers = centers.astype(np.uint8)

	# now generate the new picture
	# with NumPy arrays, arr[L] is a valid statement, and will return an array
	# of all the elements whose indices are elements of L.
	newPicture = centers[labels.flatten()]
	newPicture = newPicture.reshape(picture.shape)
	newPicture = cv2.cvtColor(newPicture, cv2.COLOR_HSV2BGR)
	# cv2.imshow("I did a (computer) science!", newPicture)
	# cv2.waitKey(0)

	return newPicture

def makeClusterPicture(image):
	"""
	Quantizes the color of an image, but instead of solving the k-means problem,
	just doens't bother	and instead clusters them around random points.
	Will create a different image every time.
	"""

	# convert the image to HSV color space.
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# cluster the hue values of the image around the three colors of the
	# Valencian flag.
	# There are four colors, but having both 0 and 180 is a workaround that
	# makes the whole thing work in the circular HSV color space.
	# hues = np.array([0, 28, 105, 180], dtype = np.int32)
	# if we want to do three random colors instead of the colors of the
	# Valencian flag, we do this instead.
	hues = np.random.randint(low = 0, high = 180, size = 3)
	print hues
	imageHSV[:, :, 0] = cluster(imageHSV[:, :, 0], hues)

	# next cluster the saturations. Unlike the hues, these aren't around any
	# specific point, just chosen randomly.
	# decide on a number of different saturation points to have.
	numSaturations = 3
	# recall that np.random.randint is on the interval [low,high)
	saturations = range(0, 256, 256 / numSaturations)
	imageHSV[:, :, 1] = cluster(imageHSV[:, :, 1], saturations)

	# # Now cluster the values in the same way as the saturations.
	numValues = 10
	values = range(0, 256, 256 / numSaturations)
	imageHSV[:, :, 2] = cluster(imageHSV[:, :, 2], values)

	# Convert back to BGR color space, or everything will look weird shades of
	# yellow.
	return cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)


def findColors():
	"""
	finds the HSV and RGB colors you're interested in
	"""
	# create 28 x 28 swatches of red, blue, and yellow.
	red = np.ones((28, 28, 3), dtype = "uint8") * np.array([0, 0, 255], dtype = "uint8")
	yellow = np.ones((28, 28, 3), dtype = "uint8") * np.array([0x0E, 0xEA, 0xFB], dtype = "uint8")
	blue = np.ones((28, 28, 3), dtype = "uint8") * np.array([0xFF, 0x7F, 0], dtype = "uint8")

	# combine the three swatches together into one image.
	flagColors = np.concatenate([red, yellow, blue], axis = 1)
	# convert the image to HSV
	flagHSV = cv2.cvtColor(flagColors, cv2.COLOR_BGR2HSV)

	# print out all the colors you just made.
	print "BGR colors"
	print "red = ", flagColors[0, 0]
	print "yellow = ", flagColors[0, 28]
	print "blue = ", flagColors[0, 56]

	print

	print "HSV colors"
	print "red = ", flagHSV[0, 0]
	print "yellow = ", flagHSV[0, 28]
	print "blue = ", flagHSV[0, 56]

	# show off your creations.
	cv2.imshow("flag", flagColors)
	cv2.waitKey(0)

	# note that imshow() will treat your images as BGR, even if they are HSV,
	# so none of the color will remotely resemble what you were expecting.
	# cv2.imshow("flag", flagHSV)
	# cv2.waitKey(0)

	# politely clean up the mess you made.
	cv2.destroyAllWindows()
	return


def main():

	original = cv2.imread("smallAntarctica.jpg")
	science = original.copy()
	#
 	# # findColors()
	# # # print image[20, 20]

	# science = playWithHue(science)
	# science = makeKmeansPicture(science)
	# science = playWithBrightness(science)
	# science = increaseSaturation(science, 1.8)
	# science = cv2.GaussianBlur(science, ksize = (7, 7), sigmaX = 1.5)
	science = increaseValue(science)


	cv2.imshow("Original", original)
	cv2.imshow("I did a (computer) science!", science)
	k= cv2.waitKey(0)

	cv2.destroyAllWindows()

	if k == 115:
		filename = raw_input("enter the filename: ")
		cv2.imwrite(filename, science)

	return

if __name__ == "__main__":
	main()
