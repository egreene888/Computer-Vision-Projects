"""
Part of the Leefsnap project, a knock-off of the Leafsnap app

(www.leafsnap.com).
Written by Evan Greene,
Credit to Matt Zucker, Mollie Wild, and the LeafSnap team.
"""
import numpy as np
import cv2
import cvk2

NUM_POINTS = 100

class leaf(object):
    def __init__(self, id, filename, species=None):
        """
        Constructor loads the image from file but does not perform any analysis
        Arguments --
            id-- the -digit code associated with the
            filename-- the location of the image
            species (optional) -- the genus and species (Latin name) of the leaf
        """
        self.id = int(id)
        self.image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self.mask = None
        self.contour = None
        self.points = None
        # identifying characteristics
        self.species = None

        if self.image is None:
            self.image = None
            self.id = None
            raise LeafError( 'Failed to open file {}'.format(filename))

    def showImage(self, caption = 'win'):
        """ helper function to show the image. Calls cv2.imshow() and shows the
        image until a key is pressed.
        Returns false if it fails, true otherwise"""
        if self.image is None:
            return False
        cv2.imshow(caption, self.image)
        while cv2.waitKey() < 15: pass

        return True

    def showMask(self, caption = 'win'):
        """ helper function to show the mask. Calls cv2.imshow() and shows the
        mask until a key is pressed
        Returns false if it fails, true otherwise"""
        if self.mask is None:
            # print("there's no mask")
            return False
        cv2.imshow(caption, self.mask)
        while cv2.waitKey() < 15: pass
        return True

    def showContour(self, caption = 'win'):
        """ function to show the contours
        Returns false if it fails, true otherwise"""
        if self.contour is None:
            return False
        size = 200
        canvas = np.zeros((2*size, 2*size))
        displayContours = self.contour + [[[size, size]]]
        cv2.fillPoly(img = canvas, pts = displayContours, color = 255)

        cv2.imshow(caption, canvas)
        while cv2.waitKey() < 15: pass

        return True

    def showPoints(self, caption = 'win'):
        if self.points is None:
            return False
        size = 200
        canvas = np.zeros((2*size, 2*size))
        for point in self.points:
            point = (int(point[0]) + size, int(point[1]) + size)
            cv2.circle(img = canvas, center = point, radius = 2,
                color = 255)
        cv2.imshow(caption, canvas)
        while cv2.waitKey() < 15: pass

        return True

    def segmentImage(self):
        # segment the image
        _ , mask = cv2.threshold(src = self.image, thresh = 0,
            maxval = 255, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Open (erode then dilate) the image
        mask = cv2.morphologyEx(src = mask, op = cv2.MORPH_CLOSE,
            kernel = (3, 3))
        self.mask = mask

    def findContours(self):
        """ Finds the largest contour (excluding the ones in the lower right)
        and returns fifty points along its length """
        if self.mask is None:
            self.segmentImage()
        # because the test images include a scale, we need to eliminate
        # images in the lower-right part of the image
        max_x = int(0.7 * self.image.shape[0])
        max_y = int(0.7 * self.image.shape[1])

        contours, hierarchy = cv2.findContours(image = self.mask, mode =
            cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

        maxAreaMoments = {'area': 0.0}
        maxAreaIndex = 0
        for i in range(len(contours)):
            m = cvk2.getcontourinfo(contours[i])
            mean = m['mean'].flatten()
            if mean[0]<=max_x and mean[1]<=max_y:
                if maxAreaMoments['area'] < m['area']:
                    maxAreaMoments = m
                    maxAreaIndex = i


        if maxAreaMoments['area'] < 5: # The segmentation Failed
            raise LeafError("""Segmentation failed for
            image {} """.format(self.id))
        biggestContour = contours[maxAreaIndex]
        # move the contour so that its center is the origin.
        biggestContour = biggestContour - maxAreaMoments['mean']
        # rotate the contour so that it's principal axis is horizontal
        angle = np.arctan2(maxAreaMoments['b1'][1], maxAreaMoments['b1'][0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        biggestContour = biggestContour.reshape((-1, 2))
        biggestContour = np.dot(rotation, biggestContour.transpose())
        # finally, normalize the area
        biggestContour *= (15000 / maxAreaMoments['area'])
        self.contour = biggestContour.transpose().reshape((-1, 1, 2)
            ).astype('int32')
        indices = np.linspace(0, biggestContour.shape[1] - 1, NUM_POINTS).tolist()
        indices = [int(x) for x in indices]
        # print(biggestContour.shape)
        # print(indices)
        self.points = np.array([ [biggestContour[0][i], biggestContour[1][i] ]
            for i in indices])
        self.points.sort(0)
        # self.showPoints()
        # self.showContour("Look! I segmented an image!")

    def test(self):
        """
        Calls each of the methods of the object to see if they work.
        """
        self.showImage(self.species)
        self.findContours()
        self.showMask(self.species)
        self.showContour(self.species)
        self.showPoints()

        return
class LeafError(Exception):
    """ a generic error class for things going wrong with the leaf """
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
