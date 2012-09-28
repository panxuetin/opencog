#include <highgui.h>
#include <iostream>
//#include <imgproc\imgproc_c.h>
#include <stdio.h>
#include "m_opencv.h"
int main(int argc, char** argv)
{

    if (argc != 2) {

        printf("Usage: %s <image>\n", argv[0]);
        return 1;
    }

    IplImage* img =  cvLoadImage(argv[1], 1);
    get_rgbs(img);
//    cv::Mat mtx(img);

    /* create a window and install mouse handler */
    cvNamedWindow("gray_img", 1);

//    cvShowImage("img",get_gray_image(img));
    cvShowImage("img",img);
//    int *grays = image_gray_values(img);
//    for (int i = 0; i < img->width * img->height; i++) {
//        std::cout<<grays[i]<<std::endl;
//    }
    output_info_img(get_gray_image(img));
    cvWaitKey(0);

    /* be tidy */
    cvDestroyAllWindows();
    cvReleaseImage(&img);
        return 0;
}
