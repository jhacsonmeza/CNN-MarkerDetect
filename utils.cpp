#include "utils.hpp"

#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

void resize_im_bbox(cv::InputOutputArray _im, cv::InputOutputArray _target, cv::Size output_size)
{
    // Resize image
    cv::Size input_size = _im.getMat().size();
    cv::resize(_im, _im, output_size);
    

    // Get labels
    cv::Mat target = _target.getMat(); //(1,n) -- (1,n,2) Vec2

    float sx = static_cast<float>(output_size.width)/input_size.width, sy = static_cast<float>(output_size.height)/input_size.height;
    int n = target.cols/2;
    float* ptarget = target.ptr<float>();
    for (int i = 0; i < n; i++)
    {
        ptarget[2*i] = sx*ptarget[2*i];
        ptarget[2*i+1] = sy*ptarget[2*i+1];
    }
}

void hflip_im_bbox(cv::InputOutputArray _im, cv::InputOutputArray _target)
{
    int w = _im.getMat().size().width;
    cv::flip(_im, _im, 1);

    cv::Mat target = _target.getMat(); //(1,n)
    target.at<float>(0) = w - target.at<float>(0);
}

void vflip_im_bbox(cv::InputOutputArray _im, cv::InputOutputArray _target)
{
    int h = _im.getMat().size().height;
    cv::flip(_im, _im, 0);

    cv::Mat target = _target.getMat(); //(1,n)
    target.at<float>(1) = h - target.at<float>(1);
}

void shift_im_bbox(cv::InputOutputArray _im, cv::InputOutputArray _target, float cx, float cy)
{
    // Get image
    cv::Mat im = _im.getMat();
    int h = im.rows, w = im.cols;

    // Get target
    cv::Mat target = _target.getMat(); //(1,n)
    float* ptarget = target.ptr<float>();
    
    // Check if translation put target out of the image
    cx = (ptarget[0]+cx > im.cols) || (ptarget[0]+cx < 0) ? -cx : cx;
    cy = (ptarget[1]+cy > im.rows) || (ptarget[1]+cy < 0) ? -cy : cy;

    // Create matrix and apply transformation
    cv::Mat M = (cv::Mat_<float>(2,3) << 1, 0, cx, 0, 1, cy);
    cv::warpAffine(_im, _im, M, im.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Get new parameters
    float xc = ptarget[0]+cx, yc = ptarget[1]+cy, wb = ptarget[2], hb = ptarget[3];
    
    // Bouding correction
    if (xc+wb/2 > w)
    {
        float x1 = xc-wb/2; // left bbox side coord
        float x2 = w; // right bbox side coord

        wb = x2-x1;
        xc = x1+wb/2;
    }

    if (xc-wb/2 < 0)
    {
        wb = xc+wb/2;
        xc = wb/2;
    }

    if (yc+hb/2 > h)
    {
        float y1 = yc-hb/2; // upper bbox side coord
        float y2 = h; // lower bbox side coord
        
        hb = y2-y1;
        yc = y1+hb/2;
    }

    if (yc-hb/2 < 0)
    {
        hb = yc+hb/2;
        yc = hb/2;
    }

    ptarget[0] = xc;
    ptarget[1] = yc;
    ptarget[2] = wb;
    ptarget[3] = hb;
}

void scale_im_bbox(cv::InputOutputArray _im, cv::InputOutputArray _target, float scale)
{
    // Get image
    cv::Mat im = _im.getMat();
    int h = im.rows, w = im.cols;

    // Get target
    cv::Mat target = _target.getMat(); //(1,n)
    float* ptarget = target.ptr<float>();

    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(ptarget[0],ptarget[1]), 0.0, scale);
    cv::warpAffine(_im, _im, M, im.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    float xc = ptarget[0]*scale + static_cast<float>(M.at<double>(0,2));
    float yc = ptarget[1]*scale + static_cast<float>(M.at<double>(1,2));
    float wb = ptarget[2]*scale;
    float hb = ptarget[3]*scale;
    
    if (xc+wb/2 > w)
    {
        float x1 = xc-wb/2; // left bbox side coord
        float x2 = w; // right bbox side coord

        wb = x2-x1;
        xc = x1+wb/2;
    }

    if (xc-wb/2 < 0)
    {
        wb = xc+wb/2;
        xc = wb/2;
    }

    if (yc+hb/2 > h)
    {
        float y1 = yc-hb/2; // upper bbox side coord
        float y2 = h; // lower bbox side coord
        
        hb = y2-y1;
        yc = y1+hb/2;
    }

    if (yc-hb/2 < 0)
    {
        hb = yc+hb/2;
        yc = hb/2;
    }

    ptarget[0] = xc;
    ptarget[1] = yc;
    ptarget[2] = wb;
    ptarget[3] = hb;
}

void contrast_brightness(cv::InputOutputArray _im, float alpha, float beta, float gamma)
{
    if (alpha != 1.f || beta != 0.f)
    {
        cv::Mat im = _im.getMat();
        uchar* pim = im.ptr(); 
        for (int i = 0; i < im.total(); i++)
            pim[i] = cv::saturate_cast<uchar>(alpha*pim[i] + beta);
    }

    if (gamma != 1)
    {
        cv::Mat table(1, 256, CV_8U);
        uchar* ptable = table.ptr();
        for (int i = 0; i < 256; i++)
            ptable[i] = cv::saturate_cast<uchar>(255.f*pow(i/255.f, 1/gamma));
        
        LUT(_im, table, _im);
    }
}

void transformer(cv::InputOutputArray _im, cv::InputOutputArray _target, float p, const cv::Size& sz)
{
    cv::RNG rng(cv::getTickCount());

    if (rng.uniform(0.f, 1.f) < p)
        hflip_im_bbox(_im, _target);
    
    if (rng.uniform(0.f, 1.f) < p)
        vflip_im_bbox(_im, _target);
    
    if (rng.uniform(0.f, 1.f) < p)
    {
        float rnd_coff = rng.uniform(-1.f, 1.f);
        float cx = cvRound(sz.width*0.06*rnd_coff);
        float cy = cvRound(sz.height*0.06*rnd_coff);

        shift_im_bbox(_im, _target, cx, cy);
    }

    if (rng.uniform(0.f, 1.f) < p)
    {
        float scale = rng.uniform(0.8f, 1.5f);
        scale_im_bbox(_im, _target, scale);
    }

    if (rng.uniform(0.f, 1.f) < p*10)
    {
        float alpha = 1.f; //rng.uniform(0.5f, 1.5f)
        float beta = 0.f; //rng.uniform(0.f, 10.f)
        float gamma = rng.uniform(0.2f, 2.f);

        contrast_brightness(_im, alpha, beta, gamma);
    }
}

// float iou_bbox(const float* pbbox1, const float* pbbox2, const int64_t* sz)
// {
//     int n = sz[0]; //Batch size
//     float metric = 0.f;
//     for (int i = 0; i < n; i++)
//     {
//         // Scale bounding boxes
//         float xc1 = pbbox1[i*4]*sz[3];
//         float yc1 = pbbox1[i*4+1]*sz[2];
//         float w1 = pbbox1[i*4+2]*sz[3];
//         float h1 = pbbox1[i*4+3]*sz[2];

//         float xc2 = pbbox2[i*4]*sz[3];
//         float yc2 = pbbox2[i*4+1]*sz[2];
//         float w2 = pbbox2[i*4+2]*sz[3];
//         float h2 = pbbox2[i*4+3]*sz[2];

//         // Estimate top-lef point
//         float x1 = std::max(xc1-w1/2, xc2-w2/2);
//         float y1 = std::max(yc1-h1/2, yc2-h2/2);

//         // Estimate bottom-right point
//         float x2 = std::min(xc1+w1/2, xc2+w2/2);
//         float y2 = std::min(yc1+h1/2, yc2+h2/2);

//         // Estimate area of intersection and area of union
//         float area_inter = std::max(0.f, x2-x1)*std::max(0.f, y2-y1);
//         float area_union = w1*h1 + w2*h2 - area_inter;

//         // Estimate IoU and add to metric
//         metric += area_inter/area_union;
//     }

//     return metric;
// }

float iou_bbox(const float* pbbox1, const float* pbbox2, const int64_t* sz)
{
    int n = sz[0]; //Batch size
    float metric = 0.f;
    for (int i = 0; i < n; i++)
    {
        // Scale bounding boxes
        float xc1 = pbbox1[i*4]*sz[3];
        float yc1 = pbbox1[i*4+1]*sz[2];
        float w1 = pbbox1[i*4+2]*sz[3];
        float h1 = pbbox1[i*4+3]*sz[2];
        cv::Rect2f bbox1(xc1-w1/2, yc1-h1/2, w1, h1);

        float xc2 = pbbox2[i*4]*sz[3];
        float yc2 = pbbox2[i*4+1]*sz[2];
        float w2 = pbbox2[i*4+2]*sz[3];
        float h2 = pbbox2[i*4+3]*sz[2];
        cv::Rect2f bbox2(xc2-w2/2, yc2-h2/2, w2, h2);

        // Estimate area of intersection and area of union
        float area_inter = (bbox1 & bbox2).area();
        float area_union = (bbox1 | bbox2).area();

        // Estimate IoU and add to metric
        metric += area_inter/area_union;
    }

    return metric;
}
