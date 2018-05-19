#pragma once
// header guard
#ifndef TRANSFORMATIONS_H
#define TRANSFORMATIONS_H

// function declarations. 

void equalizeBrightness(cv::Mat & image, unsigned char brightness = 200);

void equalizeSaturation(cv::Mat & image, unsigned char saturation = 200);

void equalizeHue(cv::Mat & image, unsigned char hue = 0);

void increaseValue(cv::Mat & image, float factor = 1.5);

void increaseSaturation(cv::Mat & image, float factor = 1.5);
#endif