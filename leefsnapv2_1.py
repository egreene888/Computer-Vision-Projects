# leefsnapv2.py

"""
An object-oriented design for LeefSnap, my copy of the LeafSnap project
(www.leafsnap.com).
Written by Evan Greene,
Credit to Matt Zucker, Mollie Wild, and the LeafSnap team.
"""
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2
import cvk2

NUM_POINTS = 100

class library(object):
    def __init__(self, size, filename):
        self.size = size
        self.file = open(filename)
        # the first line of the file is a header, so read it and do nothing.
        self.file.readline()
        self.leafPoints = np.empty([self.size, 2 * NUM_POINTS])
        self.leafInfo = np.empty([self.size, 5], dtype = "object")
        self.index = 0
        self.model = None

    def create(self):
        """ Squashes all the functionality needed to set up the library down to
        one method """
        errorcount = 0
        for _ in xrange(self.size):
            # print self.index
            try:
                self.addLeaf()
            except LeafError as e:
                 errorcount += 1
                 print e
        print "failed to segment {} images of {}.".format(errorcount, self.size)
        # print self.leafInfo[0:2]
        # print self.leafPoints[0:2]
        self.makeModel()

    def addLeaf(self):
        """ adds the contour info for an additional leaf into the library"""
        # don't overfill the library. Mainly a guard against accidentally
        # calling the function twice where we meant to call it once.
        if self.index >= self.size:
            raise LibraryError("The library is full")
        newInfo = self.readLine()
        id, image_path, segmented_path, species, kind = newInfo
        try:
            # print id
            self.leafInfo[self.index] = [id, image_path, segmented_path,
                species, kind]
            thisLeaf = leaf(id, image_path, species)
            thisLeaf.findContours()
            self.leafPoints[self.index] = thisLeaf.points.flatten()
            del thisLeaf # get rid of it for memory management purposes
            self.index += 1
        except SegmentationError:
            # print self.index

            self.leafPoints[self.index] = np.zeros((2*NUM_POINTS))
            self.leafInfo[self.index] = [id, image_path, segmented_path,
                species, 'failed']
            self.index += 1

        # thisLeaf.showImage()
        # thisLeaf.showPoints()
        # print thisLeaf.points

    def matchLeaf(self, filename, id = 0):
        """ takes a new leaf and compares it to the library. Returns a the
        string of the name of the species that it ought to be. """
        newLeaf = leaf(id = id, filename = filename,
            species = "Unknown")
        newLeaf.findContours()
        # print self.leafPoints.shape
        # print newLeaf.points.shape
        species = self.model.predict([newLeaf.points.flatten()])
        # print "indices = \n", indices[0][0]
        # print "distances = \n", distances[0][0]

        return species

    def readLine(self):
        nextline = self.file.readline()
        L = nextline.split('\t')
        return L

    def makeModel(self):
        if self.model:
            return
        sizeHiddenLayers = 650 # number of species? It's guesswork.
        self.model = MLPClassifier(solver='adam', alpha=1e-2,
            hidden_layer_sizes=(sizeHiddenLayers, sizeHiddenLayers,
            sizeHiddenLayers),
            random_state=1, max_iter = 50000, activation = 'tanh')
        self.model.fit(self.leafPoints, self.leafInfo[:, 3])

    def test(self):
        """ tests the library by using every other image as a training set, and
        every other image as a test set"""
        print "starting"

        testSet = []
        correct = 0
        total = 0
        segfail = 0
        notOsage = 0

        try:
            self.load()
            for line in self.leafInfo:
                testSet.append(line)
        except IOError:
            for _ in xrange(self.size):
                # print "Added {} leaves to library".format(self.index)
                self.addLeaf()
                testSet.append(self.readLine())
            self.save()

        print "built library with {} leaves".format(self.index)
        self.makeModel()
        print "created model, testing library"
        for elt in testSet:
            try:
                guess = self.matchLeaf(elt[1], elt[0])
                rightAnswer = elt[3]
                # print "guess -- {} ".format(guess),
                # print "Correct answer -- {}.".format(rightAnswer),
                if rightAnswer in guess:
                    correct += 1
                    # print "CORRECT!"
                # else:
                    # print "WRONG!"
                if guess != "Maclura pomifera":
                    notOsage += 1
            except LeafError:
                segfail += 1

            total += 1
        print "finished"
        print "Got {} out of {} right, {:.3f} % accuracy".format(correct, total,
            float(100 * correct) / total)
        print "Failed to segment {} out of {} ({:.3f} %)".format(segfail, total,
            float(100 * segfail) / total)
        print "There were {} that were not Osage Oranges".format(notOsage)

    def save(self):
        """ saves the important (time-intensive to generate) parts to file """
        np.savetxt("leafinfo.txt", self.leafInfo, fmt =
            ('%s', '%s', '%s', '%s', '%s'), delimiter = ',')
        np.savetxt("leafpoints.txt", self.leafPoints)

    def load(self):
        """ loads the leaf info from file """
        self.leafInfo = np.loadtxt("leafinfo.txt", dtype='str', delimiter = ',')
        # print self.leafInfo[0]
        self.leafPoints = np.loadtxt("leafpoints.txt")
        # print self.leafPoints[0].shape


class leaf(object):
    def __init__(self, id, filename, species):
        self.id = int(id)
        self.image = cv2.imread("leafsnap-dataset/" + filename,
            cv2.IMREAD_GRAYSCALE)
        self.mask = None
        self.contour = None
        self.points = None
        # identifying characteristics
        self.species = None

        if self.image is None:
            self.image = None
            self.id = None
            raise LeafError('Failed to open file {}'.format(filename))

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
            # print "there's no mask"
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
            kernel = (5, 5))
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
        for i in xrange(len(contours)):
            m = cvk2.getcontourinfo(contours[i])
            mean = m['mean'].flatten()
            if mean[0]<=max_x and mean[1]<=max_y:
                if maxAreaMoments['area'] < m['area']:
                    maxAreaMoments = m
                    maxAreaIndex = i


        if maxAreaMoments['area'] < 5: # The segmentation Failed
            raise SegmentationError("""Segmentation failed for
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
        # print biggestContour.shape
        # print indices
        self.points = np.array([ [biggestContour[0][i], biggestContour[1][i] ]
            for i in indices])
        # self.showPoints()
        # self.showContour("Look! I segmented an image!")

class LeafError(Exception):
    """ a generic error class for things going wrong with the leaf """
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class LibraryError(Exception):
    """ a generic error class for things going wrong with the library"""
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class SegmentationError(LeafError):
    """ an error class specifically for problems with segmentation """
    def __init__(self, message):
        self.message = message
    # no need for a __str__ because it will inherit the LeafError's method.

def main():
    # compile a library
    lib = library(30866/2, 'leafsnap-dataset/leafsnap-dataset-images.txt')
    lib.test()

if __name__ == "__main__":
    main()
