#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#if __has_include(<torchvision/vision.h>)
#include <torchvision/vision.h>
#else
#include "model/resnet.h"
#endif

int main()
{
    // open the first webcam plugged in the computer
    cv::VideoCapture camera(0);

    // create a window to display the images from the webcam
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    // Building model
    auto model = vision::models::ResNet18();
    model->fc = model->replace_module("fc", torch::nn::Linear(512, 4));
    torch::load(model, "../bbox++.pt", torch::kCPU);
    
    // display the frame until you press ESC
    model->eval();
    torch::NoGradGuard no_grad;
    while (camera.isOpened())
    {
        // this will contain the image from the webcam
        cv::Mat frame;
            
        // capture the next frame from the webcam
        camera >> frame;


        // Resize and normalize for target detection
        cv::Mat imr;
        cv::resize(frame, imr, cv::Size(224,224));
        imr.convertTo(imr, CV_32FC3, 1.f/255.f);

        // Convert image to torch tensor
        auto tim = torch::from_blob(imr.data, {imr.rows, imr.cols, 3}, torch::kFloat32).clone();
        tim = tim.permute({2,0,1}).unsqueeze(0);

        // Forward pass: compute predicted outputs by passing inputs to the model
        auto pred = torch::sigmoid(model->forward(tim));

        // Draw bouding box
        float* ppred = reinterpret_cast<float*>(pred.data_ptr());
        cv::Rect box(ppred[0]*frame.cols-ppred[2]*frame.cols/2, ppred[1]*frame.rows-ppred[3]*frame.rows/2, ppred[2]*frame.cols, ppred[3]*frame.rows);
        cv::rectangle(frame, box, cv::Scalar(0,0,255), 2);


        // show the image on the window
        cv::imshow("Webcam", frame);

        // ESC to break loop
        if (cv::waitKey(1) == 27)
            break;
    }
    
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
