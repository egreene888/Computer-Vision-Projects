"""
main.py

A function for testing the functionality of the leefsnap libraries
"""

import leaf

def testLeaf(folder = "/"):
    testid = 55497
    fileLoc = "dataset/images/lab/abies_concolor/ny1157-01-1.jpg"
    species = "Abies concolor"
    sample = leaf.leaf(testid, folder + fileLoc, species)
    sample.test()

def main():
    testLeaf("/home/evan/Documents/OpenCV_Projects/Computer-Vision-Projects/leefsnap/")

if __name__ == "__main__":
    main()
