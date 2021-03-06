// watercolors.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include<iostream>

// OpenCV libraries
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// project libraries
#include "transformations.h"

using namespace std; 

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		cout << "Correct usage: watercolors.cpp image_filename" << endl;
		return -1; 
	}
	// read an image in from file. 
	cv::Mat image; 
	image = cv::imread(argv[1], cv::IMREAD_COLOR);
		
	// if the image is empty, display an error message and return. 
	if (image.empty())
	{
		cout << "Could not read file " << argv[1] << endl;
		return -1; 
	}

	// otherwise, show the image. Wait for a key press. 
	cv::imshow("Window", image);
	cv::waitKey(0);

	increaseValue(image);

	cv::imshow("New Window", image);
	cv::waitKey(0);

	// clean up after ourselves by destroying the windows. 
	// Should be done automatically, but this is insurance. 
	cv::destroyAllWindows();

	return 0; 
}

