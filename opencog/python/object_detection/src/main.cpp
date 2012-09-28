#include <highgui.h>
#include <imgproc\imgproc_c.h>
#include <stdio.h>


struct LuvPixel
{
	float l;
	float u;
	float v;
	
};
RgbtoLuvPcm(imgA, width, height, luvData);

//MyLUV *luvData = new MyLUV[width*height];
IplImage* to_gray_image(const IplImage* psrc)
{
    IplImage* gray_img  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    return gray_img
}

typedef Image<RgbPixel> RgbImage;
void get_rgbs(const IplImage* psrc, int width, int height)
{
    
    RgbImage  imgR(pSrc);
}


void to_luv_image(const IplImage *pSrc, LuvPixel luvData[])
{
	long xstart=0;
	long deltapos=0;
    int width = pSrc->width;
    int height = pSrc->height;
    RgbImage img(pSrc);
	for (int i = 0; i < height; i++)
     {
		 xstart = i*width;
        for ( int j = 0; j < width; j++)
        {
			deltapos = xstart + j;
            rgb2luv((int)img[i][j].r,(int)img[i][j].g , (int)img[i][j].b ,luvData[deltapos] );
		}
	 }
}


void
mouseHandler(int event, int x, int y, int flags, void* param)
{

    IplImage* img0, * img1;
    CvFont    font;
    uchar*    ptr;
    char      label[20];

    img0 =  (IplImage*) param;
    img1 =  cvCloneImage(img0);

    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, .8, .8, 0, 1, 8);

    if (event ==   CV_EVENT_LBUTTONDOWN)
    {

        /* read pixel */
        ptr ==  cvPtr2D(img1, y, x, NULL);

        /*
         * display the BGR value
         */
        sprintf(label, "(%d, %d, %d)", ptr[0], ptr[1], ptr[2]);

        cvRectangle(
                img1,
                cvPoint(x, y - 12),
                cvPoint(x + 100, y + 4),
                CV_RGB(255, 0, 0),
                CV_FILLED,
                8, 0
                );

        cvPutText(
                img1,
                label,
                cvPoint(x, y),
                &font,
                CV_RGB(255, 255, 0)
                );

        cvShowImage("img", img1);
    }
}
    int
main(int argc, char** argv)
{

    IplImage* img;

    /* usage: <prog_name> <image> */
    if (argc != 2) {

        printf("Usage: %s <image>\n", argv[0]);
        return 1;
    }

    /* load image */
    img =  cvLoadImage(argv[1], 1);

    /* always check */
    assert(img);

    /* create a window and install mouse handler */
    cvNamedWindow("img", 1);
    cvSetMouseCallback("img", mouseHandler, (void*)img);

    cvShowImage("img", img);

    cvWaitKey(0);

    /* be tidy */
    cvDestroyAllWindows();
    cvReleaseImage(&img);
        return 0;
}
void fn(const IplImage *img)
{

//    IplImage* img2 = NULL;
//    img2 = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
//    cvCopy(img,img2);
    int width = img->width;         //图像宽度
    int height = img->height;           //图像高度
    int depth = img->depth;         //图像位深(IPL_DEPTH_8U...)
    int channels = img->nChannels;      //图像通道数(1、2、3、4)
    int imgSize = img->imageSize;       //图像大小 imageSize = height*widthStep
}

