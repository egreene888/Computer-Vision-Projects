"""
faces.py

Written by Evan Greene on 2017-07-13

A generalization of project 2 for ENGR 027, Computer Vision,
at Swarthmore College. Uses the cvk2 module written by Prof. Matt Zucker of
Swarthmore College.

Takes two images as command-line arguments. Prompts the user to click on
matching points in the two images and then uses this information to create a
blended image like the famous image that looks like Marilyn Monroe from a
distance and like Albert Einstein up close.
"""
import sys
import cv2
import numpy as np
import cvk2

def getPoints(image,filename=None ):

    win = 'Select Points by right-clicking'
    cv2.namedWindow(win)
    widget = cvk2.MultiPointWidget()

    # try to load the points from file
    if not filename:
        while (1):
            # answer = raw_input('Load points from file? (y/n):')
            answer = 'y' # shortcut
            if answer in ['y', 'Y', 'yes', 'Yes']:
                filename = raw_input('File Name: ')
                if widget.load(filename):
                    print 'Loaded points from ', filename
                    return widget.points
                else:
                    print 'Failed to load points from ', filename
            elif answer in ['n', 'N', 'no', 'No']:
                break
    else:
        widget.load(filename)
        print 'Loaded points from ', filename
        return widget.points

    success = widget.start(win, image)

    if success:
        while (1):
            answer = raw_input('Write points to file? (y/n): ')
            if answer in ['y', 'Y', 'yes', 'Yes']:
                filename = raw_input('file name?: ')
                widget.save(filename)
                break
            elif answer in ['n', 'N', 'no', 'No']:
                break
        return widget.points.astype(np.float32).reshape(-1, 1, 2)
    else:
        print 'Error getting points'
        return

def highPass(image):
    cutoff = 5
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (image - cv2.GaussianBlur(image, (0, 0), cutoff))

def lowPass(image):
    cutoff = 10
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(image, (0, 0), cutoff)

def alignImages(image1, image2):

    filename = 'obamapoints.txt'
    points1 = getPoints(image1, filename)
    filename = 'bushpoints.txt'
    points2 = getPoints(image2, filename)

    if points1 is None:
        print 'Error getting points from image 1'
        return
    if points2 is None:
        print 'Error getting points from image 2'
        return

    # find the homography that aligns the two images
    homography, _ = cv2.findHomography(points2, points1)

    [w1, h1] = image1.shape[:-1]
    [w2, h2] = image2.shape[:-1]
    corners1 = np.array([ [0, 0], [w1, 0], [w1, h1], [0, h1] ])
    corners1 = corners1.astype(np.float32).reshape((-1, 1, 2))
    corners2 = np.array([ [0, 0], [w2, 0], [w2, h2], [0, h2] ])
    corners2 = corners2.astype(np.float32).reshape((-1, 1, 2))
    transformedCorners2 = cv2.perspectiveTransform(corners2, homography)
    # print transformedCorners2
    # The eventual tranformation requires applying the homography to image 2
    # then translate it so that all of the image is in the first quadrant.
    # Then, translate the first image to align with it.
    rectangle = cv2.boundingRect(np.vstack((corners1, transformedCorners2)))
    rectangleSize = rectangle[2:4]
    translation = np.eye(3)
    translation[0][2] -= rectangle[0]
    translation[1][2] -= rectangle[1]
    # print translation

    warpedImage1 = cv2.warpPerspective(image1, translation,
        (rectangleSize[0], rectangleSize[1]))
    warpedImage2 = cv2.warpPerspective(image2, np.dot(translation, homography),
        (rectangleSize[0], rectangleSize[1]))
    assert(warpedImage1.shape == warpedImage2.shape)
    # cv2.imshow('Warped Image 1', warpedImage1)
    # while cv2.waitKey() < 15: pass
    # cv2.imshow('Warped image 2', warpedImage2)
    # while cv2.waitKey() < 15: pass

    highFreqImage = highPass(warpedImage1)
    # cv2.imshow('High-frequency signal', highFreqImage)
    # while cv2.waitKey() < 15: pass
    # lowFreqImage = lowPass(warpedImage2)
    # cv2.imshow('Low-frequency signal', lowFreqImage)
    # while cv2.waitKey() < 15: pass

    # these factors might need to be changed for different images
    k1, k2 = 0.5, 0.8
    return (k1 * highFreqImage + k2 * lowFreqImage).astype('uint8')

def createhybrid(filename1, filename2):
    image1 = cv2.imread(filename1)
    image1 = cv2.resize(image1, (600, 800))
    image2 = cv2.imread(filename2)
    image2 = cv2.resize(image2, (600, 800))

    if image1 is None:
        print 'Could not load image {}'.format(filename1)
        return
    if image2 is None:
        print 'Could not load image {}'.format(filename2)
        return

    return alignImages(image1, image2)



def main():
    if len(sys.argv) < 3:
        print "correct usage: python faces.py filename1 filename2"
        sys.exit(0)
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    hybrid = createhybrid(filename1, filename2)

    cv2.imshow('Look on my works, ye mighty, and despair', hybrid)
    while cv2.waitKey() < 15: pass

main()
