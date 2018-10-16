#include "stdafx.h"
#include <opencv2/imgproc.hpp>


// include the iostream library for debugging purposes.
#define DEBUG FALSE

#if (DEBUG)	
#include<iostream>
using namespace std;
#endif



void equalizeBrightness(cv::Mat & image, unsigned char brightness = 200) 
{
	// check the image is in color and of type unsigned 8-bit integer. 
	assert(image.depth() == CV_8U); 
	if (image.channels() == 1)
	{
		// if the image is not in color, move it to color. 
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0); 
	}
	assert(image.channels() == 3); 

	// now, change the brightness (value) element of the image to the brightness
	// specified as an argument. 

	// convert the image to HSV color space. 
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV, 0);

	// create an iterator to loop through the image. 
	cv::MatIterator_<cv::Vec3b> iterator, end;
	// loop through each pixel in the image. 
	for (iterator = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>();
		iterator != end; iterator++)
	{
		// change the brightness value of the pixel to the value given 
		(*iterator)[2] = brightness; 
	}
	
	// switch back to RGB color space
	cv::cvtColor(image, image, cv::COLOR_HSV2BGR, 0); 

	return;

}

void equalizeSaturation(cv::Mat & image, unsigned char saturation = 200)
{
	// check the image is in color and of type unsigned 8-bit integer. 
	assert(image.depth() == CV_8U);
	if (image.channels() == 1)
	{
		// if the image is not in color, move it to color. 
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0);
	}
	assert(image.channels() == 3);

	// now, change the saturation element of the image to the saturation
	// specified as an argument. 

	// convert the image to HSV color space. 
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV, 0);

	// create an iterator to loop through the image. 
	cv::MatIterator_<cv::Vec3b> iterator, end;
	// loop through each pixel in the image. 
	for (iterator = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>();
		iterator != end; iterator++)
	{
		// change the saturation value of the pixel to the value given 
		(*iterator)[1] = saturation;
	}

	// switch back to RGB color space
	cv::cvtColor(image, image, cv::COLOR_HSV2BGR, 0);

	return;
}

void equalizeHue(cv::Mat & image, unsigned char hue = 0)
{
	// check the image is in color and of type unsigned 8-bit integer. 
	assert(image.depth() == CV_8U);
	if (image.channels() == 1)
	{
		// if the image is not in color, move it to color. 
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0);
	}
	assert(image.channels() == 3);

	// now, change the saturation element of the image to the hue
	// specified as an argument. It defaults to red

	// convert the image to HSV color space. 
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV, 0);

	// create an iterator to loop through the image. 
	cv::MatIterator_<cv::Vec3b> iterator, end;
	// loop through each pixel in the image. 
	for (iterator = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>();
		iterator != end; iterator++)
	{
		// change the hue value of the pixel to the value given 
		(*iterator)[0] = hue;
	}

	// switch back to BGR color space
	cv::cvtColor(image, image, cv::COLOR_HSV2BGR, 0);

	return;
}

void increaseSaturation(cv::Mat & image, float factor = 1.5)
{	
	// check to see that the image is in color and unsigned 8-bit type. 
	// check the image is in color and of type unsigned 8-bit integer. 
	assert(image.depth() == CV_8U);
	if (image.channels() == 1)
	{
		// if the image is not in color, move it to color. 
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0);
	}
	assert(image.channels() == 3);

	// convert the image to HSV color space, in place.  
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV, 0); 

	// next, increase the saturation layer. 
	// create an iterator for the image
	cv::MatIterator_<cv::Vec3b> iterator, end;
	// loop through each pixel of the image. 
	for (iterator = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>();
		iterator != end; iterator++)
	{
		// multiply the saturation by the factor given as an argument
		int newSaturation = (*iterator)[1] * factor;
		// we need to clip the value to 255, to prevent overflow. 
		if (newSaturation > 255)
		{
			(*iterator)[1] = 255;
		}
		else
		{
			(*iterator)[1] = newSaturation;
		}
	} // end for 

	// now return the image to BGR color space. 
	cv::cvtColor(image, image, cv::COLOR_HSV2BGR, 0);
	return; 
}

void increaseValue(cv::Mat & image, float factor = 1.5)
{
	// check to see that the image is in color and unsigned 8-bit type. 
	// check the image is in color and of type unsigned 8-bit integer. 
	assert(image.depth() == CV_8U);
	if (image.channels() == 1)
	{
		// if the image is not in color, move it to color. 
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR, 0);
	}
	assert(image.channels() == 3);

	// convert the image to HSV color space, in place.  
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV, 0);

	// next, increase the saturation layer. 
	// create an iterator for the image
	cv::MatIterator_<cv::Vec3b> iterator, end;
	// loop through each pixel of the image. 
	for (iterator = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>();
		iterator != end; iterator++)
	{
		// multiply the saturation by the factor given as an argument
		int newSaturation = (*iterator)[2] * factor;
		// we need to clip the value to 255, to prevent overflow. 
		if (newSaturation > 255)
		{
			(*iterator)[2] = 255;
		}
		else
		{
			(*iterator)[2] = newSaturation;
		}
	} // end for 

	  // now return the image to BGR color space. 
	cv::cvtColor(image, image, cv::COLOR_HSV2BGR, 0);
	return;
}