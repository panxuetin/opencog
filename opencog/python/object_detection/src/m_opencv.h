#ifndef M_OPENCV.H

#define M_OPENCV.H

#include <highgui.h>

template<class T> class Image
{
    private:
        IplImage* imgp;
    public:
        Image(IplImage* img=0):imgp(img) {}
        ~Image(){imgp=0;}
        void operator=(IplImage* img) {imgp=img;}
        inline T* operator[](const int rowIndx) {
            return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
}; 

typedef struct{
    unsigned char b,g,r;
} RgbPixel; 

typedef struct{
    float b,g,r;
} RgbPixelFloat; 
typedef Image<RgbPixel> RgbImage;
//    RgbImage  imgR(pSrc);
struct LuvPixel
{
    float l;
    float u;
    float v;

};

void      rgb2luv(int R,int G, int B, LuvPixel* luvdata);
IplImage* get_gray_image(const IplImage* psrc);
LuvPixel* get_luv_image(const IplImage *pSrc);
int* image_gray_values(const IplImage *pSrc);
void output_info_img(const IplImage *img);
void get_rgbs(IplImage* pSrc);

//void get_rgbs(const IplImage* psrc, int width, int height);
#endif /* end of include guard: M_OPENCV.H */
