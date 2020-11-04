#pragma once

#include <opencv2/opencv.hpp>

void resize_im_bbox(cv::InputOutputArray, cv::InputOutputArray, cv::Size);
void hflip_im_bbox(cv::InputOutputArray, cv::InputOutputArray);
void vflip_im_bbox(cv::InputOutputArray, cv::InputOutputArray);
void shift_im_bbox(cv::InputOutputArray, cv::InputOutputArray, float, float);
void scale_im_bbox(cv::InputOutputArray, cv::InputOutputArray, float);
void contrast_brightness(cv::InputOutputArray, float, float, float);
void transformer(cv::InputOutputArray, cv::InputOutputArray, float, const cv::Size&);
float iou_bbox(const float*, const float*, const int64_t*);