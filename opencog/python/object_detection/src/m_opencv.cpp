#ifndef M_OPENCV.H

#define M_OPENCV.H

typedef Image<RgbPixel> RgbImage;
struct LuvPixel
{
	float l;
	float u;
	float v;
	
};

void rgb2luv(int R,int G, int B, LuvPixel* luvdata)
 {
		float rf, gf, bf;
		float r, g, b, X_, Y_, Z_, X, Y, Z, fx, fy, fz, xr, yr, zr;
		float L;
		float eps = 216.f/24389.f;
		float k = 24389.f/27.f;
		float Xr = 0.964221f;  // reference white D50
		float Yr = 1.0f;
		float Zr = 0.825211f;
		// RGB to XYZ
		r = R/255.f; //R 0..1
		g = G/255.f; //G 0..1
		b = B/255.f; //B 0..1   
		// assuming sRGB (D65)
		if (r <= 0.04045)
			r = r/12;
		else
			r = (float) pow((r+0.055)/1.055,2.4);
		if (g <= 0.04045)
			g = g/12;
		else
			g = (float) pow((g+0.055)/1.055,2.4);
		if (b <= 0.04045)
			b = b/12;
		else
			b = (float) pow((b+0.055)/1.055,2.4);
		X =  0.436052025f*r     + 0.385081593f*g + 0.143087414f *b;
		Y =  0.222491598f*r     + 0.71688606f *g + 0.060621486f *b;
		Z =  0.013929122f*r     + 0.097097002f*g + 0.71418547f  *b;
		// XYZ to Luv
		float u, v, u_, v_, ur_, vr_;			
		u_ = 4*X / (X + 15*Y + 3*Z);
		v_ = 9*Y / (X + 15*Y + 3*Z);		 
		ur_ = 4*Xr / (Xr + 15*Yr + 3*Zr);
		vr_ = 9*Yr / (Xr + 15*Yr + 3*Zr);
		yr = Y/Yr;
		if ( yr > eps )
			L =  116*pow(yr,float(1/3.0)) - 16;
		else
			L = k * yr;
		u = 13*L*(u_ -ur_);
		v = 13*L*(v_ -vr_);
		luvdata->l = (int) (2.55*L + .5);
		luvdata->u = (int) (u + .5); 
		luvdata->v = (int) (v + .5);       
} 
IplImage* get_gray_image(const IplImage* psrc)
{
    IplImage* gray_img  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    return gray_img
}

void info_img(const IplImage *img)
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

void get_rgbs(const IplImage* psrc, int width, int height)
{
    
    RgbImage  imgR(pSrc);
//    int width = pSrc->width;
//    int height = pSrc->height;
//	for (int i = 0; i < height; i++)
//     {
//		 xstart = i*width;
//        for ( int j = 0; j < width; j++)
//        {
//			deltapos = xstart + j;
//            rgb2luv((int)img[i][j].r,(int)img[i][j].g , (int)img[i][j].b ,luvData[deltapos] );
//		}
//	 }
}


LuvPixel* get_luv_image(const IplImage *pSrc)
{
	long xstart=0;
	long deltapos=0;
    int width = pSrc->width;
    int height = pSrc->height;
    LuvPixel *luvData = new[width * height]
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



#endif /* end of include guard: M_OPENCV.H */
