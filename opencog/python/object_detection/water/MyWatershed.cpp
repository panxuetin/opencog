#include "StdAfx.h"
#include "MyWatershed.h"
#define NearMeasureBias 200.0//�ж�������ɫ���Ƶ���ֵ��
#include <queue>
#include <imgproc\imgproc_c.h>
#include <math.h>
template<class T> class Image
{
  private:
  IplImage* imgp;
  public:
  Image(IplImage* img=0) {imgp=img;}
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
 
typedef Image<RgbPixel>       RgbImage;
typedef Image<RgbPixelFloat>  RgbImageFloat;
typedef Image<unsigned char>  BwImage;
typedef Image<float>          BwImageFloat;
void GetNeiInt(INT x, INT y, INT pos, INT width, INT height, INT &left, INT &right, INT &up, INT &down)//��pos���ĸ�����
{
	if(x - 1 >=0 )
		left = pos - 1;
	if(x + 1 < width)
		right = pos + 1;
	if(y - 1 >=0 )
		up = pos - width;
	if(y + 1 < height)
		down = pos + width;
}

void rgb2luv(int R,int G, int B, MyLUV &luvdata)
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
		luvdata.l = (int) (2.55*L + .5);
		luvdata.u = (int) (u + .5); 
		luvdata.v = (int) (v + .5);       
} 
 //## pMyColorSpace.RgbtoLuvPcm(data, width, height, luvData);
template<class T>
void RgbtoLuvPcm(T &img,INT width, INT height, MyLUV luvData[])
{
	LONG xstart=0;
	LONG deltapos=0;
	 for (int i=0; i<height; i++)
    {
		 xstart = i*width;
  
        for ( int j=0; j<width; j++)
        {
			deltapos = xstart + j;
        rgb2luv((int)img[i][j].r,(int)img[i][j].g , (int)img[i][j].b ,luvData[deltapos] );
		}
	 }
}
MyWatershed::MyWatershed(void)
{

}


MyWatershed::~MyWatershed(void)
{
}
void MyWatershed::WatershedSegmentVincent(IplImage* img)
//��ˮ��ָimgΪ��ɫ
{  
    int x,y,i; //ѭ��
 
    IplImage* pSrc = NULL;
    pSrc = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
    cvCopy(img,pSrc);
    int width = pSrc->width;         //ͼ����
    int height = pSrc->height;           //ͼ��߶�
    int depth = pSrc->depth;         //ͼ��λ��(IPL_DEPTH_8U...)
    int channels = pSrc->nChannels;      //ͼ��ͨ����(1��2��3��4)
    int imgSize = pSrc->imageSize;       //ͼ���С imageSize = height*widthStep
    int step = pSrc->widthStep/sizeof(uchar);   
    uchar* data    = (uchar *)pSrc->imageData;


    int imageLen = width*height;   
/*
    cvNamedWindow("src");f##
    cvShowImage("src",pSrc);
    */
     
    /*
    //===================ͨ��Sobel��ȡ�ݶ�ͼ��====================//
    //  ��ɫת�Ҷ�
    IplImage* gray  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    cvNamedWindow("gray");
    cvShowImage("gray",gray);
     
    //  Sobel
    IplImage* sobel = cvCreateImage(cvGetSize(pSrc),IPL_DEPTH_16S,1); //IPL_DEPTH_16S: ��Sobel��ʽ���굼������и�ֵ�����л����255��ֵ
    cvSobel(gray,sobel,1,1,3);
    IplImage *sobel8u=cvCreateImage(cvGetSize(sobel),IPL_DEPTH_8U,1);
    cvConvertScaleAbs(sobel,sobel8u,1,0); //תΪ8λ�޷���
     
    //  ��ʱ��ʾsobel
    cvNamedWindow("sobel8u");
    cvShowImage("sobel8u",sobel8u); */
     
     
    //===================�� ��ȡԭ��ɫͼ����ݶ�ֵ(��ת�Ҷ�)===================//
    int* deltar = new INT[imageLen]; //�ݶ�ģ����
    //FLOAT* deltasita = new FLOAT[imgSize];//�ݶȽǶ����飻
    //  ��ɫת�Ҷ�
    IplImage* gray  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    //  ���ݶ�
    GetGra(gray,deltar);
    //GetGra2(sobel8u,deltar);
     
 
    //===================�� �ݶ�ֵ����===================//
    //����ͳ�Ƹ��ݶ�Ƶ��
    MyImageGraPtWatershed*  graposarr = new MyImageGraPtWatershed[imageLen];
    INT*  gradientfre = new INT[256];//ͼ���и����ݶ�ֵƵ�ʣ�
    INT*  gradientadd = new INT[257];//���ݶ�����λ�ã�
    memset( gradientfre, 0, 256*sizeof(INT));
    memset( gradientadd, 0, 257*sizeof(INT));
     
    LONG xstart, imagepos, deltapos;
    xstart = imagepos = deltapos = 0;
    for (y=0; y<height; y++)
    {
        for ( x=0; x<width; x++)
        {
            xstart = y*width;
            deltapos = xstart + x;
            if (deltar[deltapos]>255)
            {
                deltar[deltapos] = 255;
            }
            INT tempi = (INT)(deltar[deltapos]);
            gradientfre[tempi] ++;//�Ҷ�ֵƵ�ʣ�
        }
    }
     
    //ͳ�Ƹ��ݶȵ��ۼӸ��ʣ�
    INT added = 0;
    gradientadd[0] = 0;//��һ����ʼλ��Ϊ0��
    for (INT ii=1; ii<256; ii++)
    {
        added += gradientfre[ii-1];
        gradientadd[ii] = added;
    }
    gradientadd[256] = imageLen;//���λ�ã�
     
    memset( gradientfre, 0, 256*sizeof(INT)); //���㣬��������ĳ�ݶ��ڵ�ָ�룻
     
    //������������sorting....
    for (y=0; y<height; y++)
    {
        xstart = y*width;
        for ( x=0; x<width; x++)
        {
            deltapos = xstart + x;
            INT tempi = (INT)(deltar[deltapos]);//��ǰ����ݶ�ֵ������ǰ��Ĳ��裬���ֻ��Ϊ255��
            //�����ݶ�ֵ���������������е�λ�ã�
            INT tempos = gradientadd[tempi] + gradientfre[tempi];
            gradientfre[tempi] ++;//�ݶ���ָ����ƣ�
            graposarr[tempos].gradient = tempi; //���ݵ�ǰ����ݶȽ��õ���Ϣ�ź�����������еĺ���λ����ȥ��    
            graposarr[tempos].x = x;
            graposarr[tempos].y = y;
        }
    }
 
 
    //===================�� �����ÿ����������������flag=================//
    INT rgnumber = 0;                   //�ָ���������
    INT*   flag = new INT[imageLen];        //�����ʶ����
    Flood(graposarr,gradientadd,0,255,flag,rgnumber,width,height); //!rgnumber = the number of area 
	//printf("%d******************%d",rgnumber,imageLen);

	//for(int i=0;i<imageLen;i++)
	{
		//printf("%d*",flag[i]);
		//if(!(i%90))
		//	printf("\n");
	}
     
     
    //===================�� ��flag�������������LUV��ֵ=================//
    //����׼��������������LUV��ֵ��
    //�ָ���������һЩͳ����Ϣ,��һ��Ԫ�ز��ã�ͼ���и��������������Ϣ�����flag������
    MyRgnInfoWatershed*  rginfoarr = new MyRgnInfoWatershed[rgnumber+1];   
    //��ո�����
    for (i=0; i<rgnumber+1; i++) //LUV
    {
        rginfoarr[i].isflag = FALSE;
        rginfoarr[i].ptcount = 0;
        rginfoarr[i].l = 0;
        rginfoarr[i].u = 0;
        rginfoarr[i].v = 0;
    }
     
    //���±���LUV���ݣ�
	

	
    MyLUV *luvData = new MyLUV[width*height];
   //## pMyColorSpace.
	RgbImage  imgA(pSrc);
	RgbtoLuvPcm(imgA, width, height, luvData);
 
    for (y=0; y<height; y++)
    {
        xstart = y*width;
        for ( x=0; x<width; x++)
        {
            INT pos = xstart + x;
            INT rgid = flag[pos];//��ǰλ�õ�������������ͳ����Ϣ�����е�λ��
            //���½��õ����Ϣ�ӵ�����������Ϣ��ȥ
            rginfoarr[rgid].ptcount ++;
            rginfoarr[rgid].l += luvData[pos].l;
            rginfoarr[rgid].u += luvData[pos].u;
            rginfoarr[rgid].v += luvData[pos].v;
        }
    }
 
    //�����������LUV��ֵ��
    for (i=0; i<=rgnumber; i++)
    {
        rginfoarr[i].l = (FLOAT) ( rginfoarr[i].l / rginfoarr[i].ptcount );
        rginfoarr[i].u = (FLOAT) ( rginfoarr[i].u / rginfoarr[i].ptcount );
        rginfoarr[i].v = (FLOAT) ( rginfoarr[i].v / rginfoarr[i].ptcount );
    }
 
    //===================�� �����ں� ===================//
    //���ݸ���luv��ֵ��rginfoarr���͸���֮���ڽӹ�ϵ����flag���㣩���������ں�
    INT* mergearr = new INT[rgnumber+1];
    memset( mergearr, -1, sizeof(INT)*(rgnumber+1) );
    INT mergednum = 0;
    //entrance
    MergeRgs(rginfoarr, rgnumber, flag, width, height, mergearr, mergednum);
    //MergeRgs(rginfoarr, rgnumber, flag, width, height);
     
 
    //===================�� ȷ���ںϺ����������������===================//
    //ȷ���ϲ�������ص���������
    for (y=0; y<(height); y++)
    {
        xstart = y*width;
        for ( x=0; x<(width); x++)
        {
            INT pos = xstart + x;
            INT rgid = flag[pos];//�õ���������
            flag[pos] = FindMergedRgn(rgid, mergearr);
        }
    }
    delete [] mergearr;
    mergearr = NULL;
 
    //===================�� ��������������������===================//
    //=================== ===================//
    //���¸��ݱ�ʶ������ұ߽�㣨�����ıߵ��Լ�С���Ӷȣ�
    IplImage* dstContour = cvCreateImage(cvGetSize(pSrc),IPL_DEPTH_8U,3);
 
    for (y=1; y<(height-1); y++)
    {
        xstart = y*width;
        for ( x=1; x<(width-1); x++)
        {
            INT pos = xstart + x;
            INT imagepos = pos * 3;
            INT left = pos - 1;
            INT up = pos - width;
            INT right = pos +1;
            INT down = pos + width;
            if ( ( flag[right]!=flag[pos] ) || ( flag[down]!=flag[pos] ) )
            //if ( flag[pos]==0 )
            {
                ((uchar *)(dstContour->imageData+y*dstContour->widthStep))[x*dstContour->nChannels+0] = 0;
                ((uchar *)(dstContour->imageData+y*dstContour->widthStep))[x*dstContour->nChannels+1] = 0;
                ((uchar *)(dstContour->imageData+y*dstContour->widthStep))[x*dstContour->nChannels+2] = 250;
                //imageData[imagepos] = 0;
                //imageData[imagepos+1] = 0;
                //imageData[imagepos+2] = 250;
            }
        }
    }
 
    cvNamedWindow("dstContour",1);
    cvShowImage("dstContour",dstContour);
 
    //=================== ===================//
    //  ��Դͼ���ϻ�������
    CvMemStorage* storage= cvCreateMemStorage(0);
    CvSeq* contour= 0;//�ɶ�̬����Ԫ������
    IplImage* dstContourGray= cvCreateImage(cvGetSize(dstContour),8,1);
    cvCvtColor(dstContour,dstContourGray,CV_BGR2GRAY);
    cvNamedWindow("dstContourGray",1);
    cvShowImage("dstContourGray",dstContourGray);
 
    cvFindContours( //����cvFindContours�Ӷ�ֵͼ���м��������������ؼ�⵽�������ĸ���
        dstContourGray,//��ֵ��ͼ��
        storage,    //�����洢����
        &contour,   //ָ���һ�����������ָ��
        sizeof(CvContour),  //����ͷ��С
        CV_RETR_CCOMP,      //�������������; CV_RETR_EXTERNAL��ֻ���������������
        CV_CHAIN_APPROX_SIMPLE//ѹ��ˮƽ��ֱ�ͶԽǷָ������ֻ����ĩ�˵����ص� CV_CHAIN_CODE//CV_CHAIN_APPROX_NONE
        );
     
    IplImage* dst = cvCreateImage(cvGetSize(pSrc),IPL_DEPTH_8U,3);
    cvCopy(pSrc,dst);
    for (;contour!= 0;contour= contour->h_next)
    {
        CvScalar color= CV_RGB(rand()&255,rand()&255,rand()&255);
        cvDrawContours( //��ͼ���л�������
            dst,
            contour,//ָ���ʼ������ָ��
            color,  //�����������ɫ
            color,  //�ڲ���������ɫ
            -1,     //�������������ȼ���
                        //0����ǰ��
                        //1����ͬ�����µ�������
                        //2������ͬ������һ��������
                        //������������Ƶ�ǰ����֮����������������������������abs(�ò���)-1��Ϊֹ��
            1,      //����������ϸ
            8       //����������
            );
    }
    cvNamedWindow("Contours",1);
    cvShowImage("Contours",dst);
    cvSaveImage("D:\\result.jpg",dst);
    cvReleaseImage(&dstContour);
    cvReleaseImage(&dst);
    cvReleaseImage(&dstContourGray);
    cvReleaseMemStorage(&storage);
 
    cvWaitKey(0);
    cvDestroyAllWindows();
 
    //  �ͷ���Դ
    delete [] gradientadd; gradientadd = NULL;//��С256
    delete [] gradientfre; gradientfre = NULL;//��С256
    delete [] deltar;    deltar    = NULL;//��СimageLen
    delete [] graposarr; graposarr = NULL;//��СimageLen
	    if ( luvData != NULL )
    {
        delete [] luvData;
        luvData = NULL;
    }
    cvReleaseImage(&pSrc);
    cvReleaseImage(&gray); 
	printf("\nDone\n");
     
 
}
//*****************************************************************************************************************
void MyWatershed::GetGra(IplImage* image, INT* deltar)
//ͨ��sobel��ȡ�Ҷ�ͼ��image���ݶ�ֵ������deltar
{
    //   ����
    IplImage* pSrc = NULL;
    pSrc = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
    cvCopy(image,pSrc);
    int width = pSrc->width;
    int height = pSrc->height;
 
/*  ���Բ���
    CString mPath;
    //m_Path.GetWindowText(mPath);
    mPath.Format(_T("width:%d, height: %d"),width,height);
    AfxMessageBox(mPath);
    */
 
    static int nWeight[2][3][3];
    nWeight[0][0][0]=-1;
    nWeight[0][0][1]=0;
    nWeight[0][0][2]=1;
    nWeight[0][1][0]=-2;
    nWeight[0][1][1]=0;
    nWeight[0][1][2]=2;
    nWeight[0][2][0]=-1;
    nWeight[0][2][1]=0;
    nWeight[0][2][2]=1;
    nWeight[1][0][0]=1;
    nWeight[1][0][1]=2;
    nWeight[1][0][2]=1;
    nWeight[1][1][0]=0;
    nWeight[1][1][1]=0;
    nWeight[1][1][2]=0;
    nWeight[1][2][0]=-1;
    nWeight[1][2][1]=-2;
    nWeight[1][2][2]=-1;
     
    int nTmp[3][3];
    FLOAT dGray;
    FLOAT dGradOne;
    FLOAT dGradTwo;
    int x,y;
    int* gra=new int[height*width];
    int gr;
    for (y=0;y<height;y++)
    {
        for (x=0;x<width;x++)
        {
            int image_pos=width*y+x;
            /*
            int memory_pos1=3*width*y+3*x;
            int memory_pos2=3*width*y+3*x+1;
            int memory_pos3=3*width*y+3*x+2;
            FLOAT b=image[memory_pos1];
            FLOAT g=image[memory_pos2];
            FLOAT r=image[memory_pos3];
            int gr=(int)(0.299*r+0.587*g+0.114*b);*/
            gr = ((uchar *)(pSrc->imageData))[y*pSrc->widthStep+x];
            gra[image_pos]=gr;
        }
    }
 
    //�ݲ������Ե��
    for (y=1;y<height-1;y++)
    {
        for (x=1;x<width-1;x++)
        {
            dGray=0;
            dGradOne=0;
            dGradTwo=0;
            nTmp[0][0]=gra[(y-1)*width+x-1];
            nTmp[0][1]=gra[(y-1)*width+x];
            nTmp[0][2]=gra[(y-1)*width+x+1];
            nTmp[1][0]=gra[y*width+x-1];
            nTmp[1][1]=gra[y*width+x];
            nTmp[1][2]=gra[y*width+x+1];
            nTmp[2][0]=gra[(y+1)*width+x-1];
            nTmp[2][1]=gra[(y+1)*width+x];
            nTmp[2][2]=gra[(y+1)*width+x+1];
            for (int yy=0;yy<3;yy++)
            {
                for (int xx=0;xx<3;xx++)
                {
                    dGradOne+=nTmp[yy][xx]*nWeight[0][yy][xx];
                    dGradTwo+=nTmp[yy][xx]*nWeight[1][yy][xx];
                }
            }
            dGray=dGradOne*dGradOne+dGradTwo*dGradTwo;
            dGray=sqrtf(dGray);
            deltar[y*width+x]=(int)dGray;
        }
    }
     
    //��Ե��Ϊ���ڲ���ֵ
    for (y=0; y<height; y++)
    {
        INT x1 = 0;
        INT pos1 = y*width + x1;
        deltar[pos1] = deltar[pos1+1]; 
        INT x2 = width-1;
        INT pos2 = y*width + x2;
        deltar[pos2] = deltar[pos2-1];
    }
    for ( x=0; x<width; x++)
    {
        INT y1 = 0;
        INT pos1 = x;
        INT inner = x + width;//��һ�У�
        deltar[pos1] = deltar[inner];
        INT y2 = height-1;
        INT pos2 = y2*width + x;
        inner = pos2 - width;//��һ�У�
        deltar[pos2] = deltar[inner];
    }
     
    delete [] gra;
    gra=NULL;
 
}
 
 
//////////////////////////////////////////////////////////////////////////
// Luc Vincent and Pierre Soille�ķ�ˮ��ָ�flood�����ʵ�ִ��룬
// �޸�����Ӧα����, α���������������ġ�Watersheds in Digital Spaces:
// An Efficient Algorithm Based on Immersion Simulations��
// IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE.
// VOL.13, NO.6, JUNE 1991;
// by dzj, 2004.06.28
// MyImageGraPt* imiarr - ��������������
// INT* graddarr -------- ����ĸ��ݶ����飬�ɴ�ֱ�Ӵ�ȡ��H���ص�
// INT minh��INT maxh --- ��С����ݶ�
// INT* flagarr --------- ����������
// ע�⣺Ŀǰ�����ˮ���ǣ�ֻ��ÿ��������������
//////////////////////////////////////////////////////////////////////////
void MyWatershed::Flood(MyImageGraPtWatershed* imiarr, INT* graddarr,
                         INT minh, INT maxh,
                         INT* flagarr, INT& outrgnumber,
                         INT width,INT height)
{
    int imageWidth=width;
    int imageHeight=height;
 
    const INT INIT = -2;//initial value of lable image
    const INT MASK = -1;//initial value at each level
    const INT WATERSHED = 0; //label of the watershed pixels
    INT h = 0;
	INT i=0;
//	INT ini=0;
    INT imagelen = imageWidth * imageHeight;//ͼ�������صĸ���
    for ( i=0; i<imagelen; i++)
    {
        flagarr[i] = INIT;
    }
    //memset(flagarr, INIT, sizeof(INT)*imagelen);
    INT* imd = new INT[imagelen]; //�������飬ֱ�Ӵ�ȡ��
    for (i=0; i<imagelen; i++)
    {
        imd[i] = 0;
    }
    //memset(imd, 0, sizeof(INT)*imagelen);
    std::queue <int> myqueue;
    INT curlabel = 0;//����ر�ǣ�
     
    for (h=minh; h<=maxh; h++)
    {
        INT stpos = graddarr[h];
        INT edpos = graddarr[h+1];
        for (INT ini=stpos; ini<edpos; ini++)
        {
            INT x = imiarr[ini].x;
            INT y = imiarr[ini].y;
            INT ipos = y*imageWidth + x;
            flagarr[ipos] = MASK;
	    //###
            //���¼��õ������Ƿ��ѱ������ĳ�����ˮ�룬���ǣ��򽫸õ����fifo;
            INT left = ipos - 1;
            if (x-1>=0)
            {
                if (flagarr[left]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//��λ��ѹ��fifo;
                    continue;
                }              
            }
            INT right = ipos + 1;
            if (x+1<imageWidth)
            {
                if (flagarr[right]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//��λ��ѹ��fifo;
                    continue;
                }
            }
            INT up = ipos - imageWidth;
            if (y-1>=0)
            {
                if (flagarr[up]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//��λ��ѹ��fifo;
                    continue;
                }              
            }
            INT down = ipos + imageWidth;
            if (y+1<imageHeight)
            {
                if (flagarr[down]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//��λ��ѹ��fifo;
                    continue;
                }          
            }
        }
         
        //���¸����Ƚ��ȳ�������չ������أ�
        INT curdist = 1; myqueue.push(-99);//�����ǣ�
        while (TRUE)
        {
            INT p = myqueue.front();
            myqueue.pop();
            if (p == -99)
            {
                if ( myqueue.empty() )
                {
                    break;
                }else
                {
                    myqueue.push(-99);
                    curdist = curdist + 1;
                    p = myqueue.front();
                    myqueue.pop();
                }
            }
             
            //������p������
            INT y = (INT) (p/imageWidth);
            INT x = p - y*imageWidth;
            INT left = p - 1;
            if  (x-1>=0)
            {
                if ( ( (imd[left]<curdist) && flagarr[left]>0)
                    || (flagarr[left]==0) )
                {
                    if ( flagarr[left]>0 )
                    {
                        //ppei����ĳ���򣨲��Ƿ�ˮ�룩��
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //������Ϊ�ڵ���������
                            flagarr[p] = flagarr[left];
                        }else if (flagarr[p]!=flagarr[left])
                        {
                            //ԭ�������������ڸ�������ͬ����Ϊ��ˮ�룻
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppeiΪ���룻
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[left]==MASK) && (imd[left]==0) )
                    //ppei����MASK�ĵ㣬����δ��ǣ�������ĳ��Ҳ���Ƿ�ˮ�룩;
                {
                    imd[left] = curdist + 1; myqueue.push(left);
                }
            }
             
            INT right = p + 1;
            if (x+1<imageWidth)
            {
                if ( ( (imd[right]<curdist) &&  flagarr[right]>0)
                    || (flagarr[right]==0) )
                {
                    if ( flagarr[right]>0 )
                    {
                        //ppei����ĳ���򣨲��Ƿ�ˮ�룩��
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //������Ϊ�ڵ���������
                            flagarr[p] = flagarr[right];
                        }else if (flagarr[p]!=flagarr[right])
                        {
                            //ԭ�������������ڸ�������ͬ����Ϊ��ˮ�룻
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppeiΪ���룻
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[right]==MASK) && (imd[right]==0) )
                    //ppei����MASK�ĵ㣬����δ��ǣ�������ĳ��Ҳ���Ƿ�ˮ�룩;
                {
                    imd[right] = curdist + 1; myqueue.push(right);
                }
            }
             
            INT up = p - imageWidth;
            if (y-1>=0)
            {
                if ( ( (imd[up]<curdist) &&  flagarr[up]>0)
                    || (flagarr[up]==0) )
                {
                    if ( flagarr[up]>0 )
                    {
                        //ppei����ĳ���򣨲��Ƿ�ˮ�룩��
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //������Ϊ�ڵ���������
                            flagarr[p] = flagarr[up];
                        }else if (flagarr[p]!=flagarr[up])
                        {
                            //ԭ�������������ڸ�������ͬ����Ϊ��ˮ�룻
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppeiΪ���룻
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[up]==MASK) && (imd[up]==0) )
                    //ppei����MASK�ĵ㣬����δ��ǣ�������ĳ��Ҳ���Ƿ�ˮ�룩;
                {
                    imd[up] = curdist + 1; myqueue.push(up);
                }
            }
             
            INT down = p + imageWidth;
            if (y+1<imageHeight)
            {
                if ( ( (imd[down]<curdist) &&  flagarr[down]>0)
                    || (flagarr[down]==0) )
                {
                    if ( flagarr[down]>0 )
                    {
                        //ppei����ĳ���򣨲��Ƿ�ˮ�룩��
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //������Ϊ�ڵ���������
                            flagarr[p] = flagarr[down];
                        }else if (flagarr[p]!=flagarr[down])
                        {
                            //ԭ�������������ڸ�������ͬ����Ϊ��ˮ�룻
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppeiΪ���룻
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[down]==MASK) && (imd[down]==0) )
                    //ppei����MASK�ĵ㣬����δ��ǣ��Ȳ���ĳ��Ҳ���Ƿ�ˮ�룩;
                {
                    imd[down] = curdist + 1; myqueue.push(down);
                }  
            }
             
        }//����������ص���չ��
         
        //���´����·��ֵ���أ�
        for (INT ini=stpos; ini<edpos; ini++)
        {
            INT x = imiarr[ini].x;
            INT y = imiarr[ini].y;
            INT ipos = y*imageWidth + x;
            imd[ipos] = 0;//�������о���
            if (flagarr[ipos]==MASK)
            {
                //����ǰ����չ��õ���ΪMASK����õ��Ϊ����ص�һ����ʼ��;
                curlabel = curlabel + 1;
                myqueue.push(ipos);
                flagarr[ipos] = curlabel;
                 
                while ( myqueue.empty()==FALSE )
                {
                    INT ppei = myqueue.front();
                    myqueue.pop();
                    INT ppeiy = (INT) (ppei/imageWidth);
                    INT ppeix = ppei - ppeiy*imageWidth;
                     
                    INT ppeileft = ppei - 1;
                    if ( (ppeix-1>=0) && (flagarr[ppeileft]==MASK) )
                    {
                        myqueue.push(ppeileft);//��λ��ѹ��fifo;
                        flagarr[ppeileft] = curlabel;
                    }
                    INT ppeiright = ppei + 1;
                    if ( (ppeix+1<imageWidth) && (flagarr[ppeiright]==MASK) )
                    {
                        myqueue.push(ppeiright);//��λ��ѹ��fifo;
                        flagarr[ppeiright] = curlabel;
                    }
                    INT ppeiup = ppei - imageWidth;
                    if ( (ppeiy-1>=0) && (flagarr[ppeiup]==MASK) )
                    {
                        myqueue.push(ppeiup);//��λ��ѹ��fifo;
                        flagarr[ppeiup] = curlabel;
                    }
                    INT ppeidown = ppei + imageWidth;
                    if ( (ppeiy+1<imageHeight) && (flagarr[ppeidown]==MASK) )
                    {
                        myqueue.push(ppeidown);//��λ��ѹ��fifo;
                        flagarr[ppeidown] = curlabel;
                    }                  
                }              
            }
        }//���ϴ����·��ֵ���أ�
         
    }
     
    outrgnumber = curlabel;
    delete [] imd;
    imd = NULL;
     
    /*
    const INT INIT = -2;
    const INT MASK = -1;
    const INT WATERSHED = 0;
    INT imagelen=width*height;
    INT *imd=new INT[imagelen];
    memset(imd, 0, sizeof(INT)*imagelen);
    std::queue <int> myqueue;
    INT curlabe=0;
    for (int i=minh;i<=maxh;i++)
    {
    INT stpos=graddarr[i];
    INT edpos=graddarr[i+1];
    INT x,y;
    static int disX[]={1,0,-1,0};
    static int disY[]={0,1,0,-1};
    for (int j=stpos;j<edpos;j++)
    {
    x=imiarr[j].x;
             y=imiarr[j].y;
             int pos=y*width+x;
             flagarr[pos]=MASK;
             for (int ii=0;ii<4;ii++)
             {
             int nepos=(y+disY[ii])*width+x+disX[ii];
             if (((x+disX[ii])>=0)&&((x+disX[ii])<width)&&
             ((y+disY[ii])>=0)&&((y+disY[ii])<height)&&(flagarr[nepos]>0||flagarr[nepos]==0))
             {     
             imd[pos]=1;
             myqueue.push(pos);
             break;
             }
             }
             }
              
               int curdist=1;//�����־,���˸о���־���Ƿ񱻴�����
               myqueue.push(-99);//������
               while (TRUE)
               {
               int siz=myqueue.size();
               int p=myqueue.front();
               myqueue.pop();
               if (p==-99)
               {
               if (myqueue.empty()==TRUE)
               {
               break;
               }
               else
               {
               myqueue.push(-99);
               curdist+=1;
               p=myqueue.front();
               myqueue.pop();
               }
               }
               //����p���ٵ�
               for(int ii=0;ii<4;ii++)
               {
               y=(INT)(p/width);
               x=imagelen-y*width;
               if (((x+disX[ii])>0||((x+disX[ii])==0))&&((x+disX[ii])<width)&&
               ((y+disY[ii])>0||((y+disY[ii])==0))&&((y+disY[ii])<height))
               {
               int npos=(y+disY[ii])*width+x+disX[ii];
               if (imd[npos]<curdist&&(flagarr[npos]>0||flagarr[npos]==0))
               {
                
                 if (flagarr[npos]>0)
                 {
                 if (flagarr[p]==MASK||flagarr[p]==WATERSHED)
                 {
                 flagarr[p]=flagarr[npos];
                 }
                 else if (flagarr[p]!=flagarr[npos])
                 {
                 flagarr[p]=WATERSHED;
                 }
                 }
                 else if (flagarr[p]==MASK)
                 {
                 flagarr[p]=WATERSHED;
                 }
                  
                   }
                   else if (flagarr[npos]==MASK&&imd[npos]==0)
                   {
                   imd[npos]=curdist+1;
                   myqueue.push(npos);
                   }                  
                   }
                   }
                   }//while
                    
                      
                       //�����ٵ��for
                       for (int jj=stpos;jj<edpos;jj++)
                       {
                       x=imiarr[jj].x;
                       y=imiarr[jj].y;
                       INT ipos = y*width + x;
                       imd[ipos] = 0;//�������о���
                       if (flagarr[ipos]==MASK)
                       {
                       //����ǰ����չ��õ���ΪMASK����õ��Ϊ����ص�һ����ʼ��;
                       curlabe = curlabe + 1;
                       myqueue.push(ipos);
                       flagarr[ipos] = curlabe;
                       while ( myqueue.empty()==FALSE )
                       {
                       INT ppei = myqueue.front();
                       myqueue.pop();
                       int s=myqueue.size();
                       INT ppeiy = (INT) (ppei/width);
                       INT ppeix = ppei - ppeiy*width;
                        
                         INT ppeileft = ppei - 1;
                         if ( (ppeix-1>=0) && (flagarr[ppeileft]==MASK) )
                         {
                         myqueue.push(ppeileft);//��λ��ѹ��fifo;
                         flagarr[ppeileft] = curlabe;
                         }
                         INT ppeiright = ppei + 1;
                         if ( (ppeix+1<width) && (flagarr[ppeiright]==MASK) )
                         {
                         myqueue.push(ppeiright);//��λ��ѹ��fifo;
                         flagarr[ppeiright] = curlabe;
                         }
                         INT ppeiup = ppei - width;
                         if ( (ppeiy-1>=0) && (flagarr[ppeiup]==MASK) )
                         {
                         myqueue.push(ppeiup);//��λ��ѹ��fifo;
                         flagarr[ppeiup] = curlabe;
                         }
                         INT ppeidown = ppei + width;
                         if ( (ppeiy+1<height) && (flagarr[ppeidown]==MASK) )
                         {
                         myqueue.push(ppeidown);//��λ��ѹ��fifo;
                         flagarr[ppeidown] = curlabe;
                         }                 
                         }             
                         }
                         }//�����ٵ��for
                          
                            
                             }//����һ��for
                              
                               outrgnumber=curlabe;
                               delete [] imd;
                               imd=NULL;
*/
}
 

//********************************************************************************* 
/**
 * @brief entrance point
 *
 * @param rginfoarr ����������Ϣ
 * @param rgnumber
 * @param flag
 * @param width
 * @param height
 * @param outmerge
 * @param rgnum
 */
void MyWatershed::MergeRgs(MyRgnInfoWatershed* rginfoarr, INT rgnumber, INT* flag, INT width, INT height, INT* outmerge, INT& rgnum)
//�ϲ���������
{
    //////////////////////////////////////////////////////////////////////////
    //1�������������������飻
    //2������ɨ�������Ѱ�Ҽ�С����
    //3����ÿ����С����A���������������ҵ��������ߣ�
    //4������������B���ϲ���������Ϣˢ�£����ڼ�С����A����������
    //   ɾ����������B����������������ɾ����������B����Ӧ�����
    //   ��������B����������s�ӵ���С����A����������ȥ��
    //5����¼�ϲ���Ϣ����һ����ר�Ŵ�Ÿ���Ϣ��������ĵ�A��Ԫ��ֵ��ΪB��
    //6���ж��Ƿ���Ϊ��С���������򷵻�3��
    //7���Ƿ����������Ѵ�����ϣ������򷵻�2��
    //
    //   ���ڸ���������������̫�࣬��˲����ڽ�������Ϊ�洢�ṹ��
    //////////////////////////////////////////////////////////////////////////
    CString* neiarr = new CString[rgnumber+1];//��һ�����ã�
    INT* mergearr = outmerge;//��¼�ϲ�������飻
 
    //�����������飻
    for (INT y=0; y<height; y++)
    {
        INT lstart = y * width;
        for (INT x=0; x<width; x++)
        {
            INT pos = lstart + x;
            INT left=-1, right=-1, up=-1, down=-1;
           //## pMyMath.
			GetNeiInt(x, y, pos, width, height, left, right, up, down);//��pos���ĸ�����
            //ȷ����ˢ����������Ϣ��
            INT curid = flag[pos];
            AddNeiOfCur(curid, left, right, up, down, flag, neiarr);
        }
    }//�����������飻
     
    //������Ϣ�����е���Ч��Ϣ��1��ʼ����i��λ�ô�ŵ�i���������Ϣ��
    for (INT rgi=1; rgi<=rgnumber; rgi++)
    {
        //ɨ�����������Ҽ�С����
        LONG allpoints = width * height;
        LONG nmin = (LONG) (allpoints / 400);
        INT curid = rgi;
 
        //rginfoarr[rgi].isflag��ʼΪFALSE���ڱ��ϲ������������ΪTRUE��
        while ( ( (rginfoarr[rgi].ptcount)<nmin )
            && !rginfoarr[rgi].isflag )
        {
            //����Ϊ��С������������������������ӽ��ߣ�
            CString neistr = neiarr[curid];
            INT nearid = FindNearestNei(curid, neistr, rginfoarr, mergearr);
            //�ϲ�curid��nearid��
            MergeTwoRgn(curid, nearid, neiarr, rginfoarr, mergearr);           
        }
    }
 
    //�����ٺϲ��������򣬣����۴�С��,�������Ҫ��ֱ�ӽ�����ѭ��ע�͵������ˣ�
    INT countjjj = 0;
    //������Ϣ�����е���Ч��Ϣ��1��ʼ����i��λ�ô�ŵ�i���������Ϣ��
    for (INT ii=1; ii<=rgnumber; ii++)
    {
        if (!rginfoarr[ii].isflag)
        {
            INT curid = ii;
            MergeNearest(curid, rginfoarr, neiarr, mergearr);
        }
    }
 
    INT counttemp = 0;
    for (INT i=0; i<rgnumber; i++)
    {
        if (!rginfoarr[i].isflag)
        {
            counttemp ++;
        }
    }
 
    rgnum = counttemp;
 
    delete [] neiarr;
    neiarr = NULL;
}
 

INT MyWatershed::FindMergedRgnMaxbias(INT idint, INT* mergearr, INT bias)
//����ֵ��ֹ���Һϲ���������coarse watershed,
//�����߱��뱣֤idint��Ч������mergearr[idint]>0��
//�Լ�mergearr��Ч������mergearr[idint]<idint;
{
    INT outid = idint;
    while ( mergearr[outid]<bias )
    {
        outid = mergearr[outid];
    }
    return mergearr[outid];
}
 
INT MyWatershed::FindMergedRgn(INT idint, INT* mergearr)
//�ҵ�idint�������ϲ��������ţ�
{
    INT outid = idint;
    while ( mergearr[outid] > 0 )
    {
        outid = mergearr[outid];
    }
    return outid;
}
 

/**
 * @brief �ϲ���������
 *
 * @param curid
 * @param rginfoarr
 * @param neiarr
 * @param mergearr
 */
void MyWatershed::MergeNearest(INT curid, MyRgnInfoWatershed* rginfoarr, CString* neiarr, INT* mergearr)
{
    //���δ���������������ƣ���ϲ���
    //CString neistr = neiarr[curid];
    FLOAT cl, cu, cv;
    cl = rginfoarr[curid].l;//��ǰ����LUVֵ��
    cu = rginfoarr[curid].u;
    cv = rginfoarr[curid].v;
    BOOL loopmerged = TRUE;//һ��ѭ�����Ƿ��кϲ��������������ޣ����˳�ѭ����
 
    while (loopmerged)
    {
        loopmerged = FALSE;
        CString tempstr = neiarr[curid];//���ڱ������ڲ�����
        while (tempstr.GetLength()>0)
        {
            INT pos = tempstr.Find(_T(' '));
            ASSERT(pos>=0);
            CString idstr = tempstr.Left(pos);
            tempstr.Delete(0, pos+1);
      //       idstr.string
          INT idint = _wtoi(idstr);
            //�жϸ����Ƿ��ѱ��ϲ������ǣ���һֱ�ҵ�������ǰ�����ţ�
            idint = FindMergedRgn(idint, mergearr);
            if (idint==curid)
            {
                continue;//��������ѱ��ϲ�����ǰ����������
            }
            FLOAT tl, tu, tv;
            tl = rginfoarr[idint].l;//��ǰ�����������LUVֵ;
            tu = rginfoarr[idint].u;
            tv = rginfoarr[idint].v;
            DOUBLE tempdis = pow(tl-cl, 2)
                + pow(tu-cu, 2) + pow(tv-cv, 2);
            if (tempdis<NearMeasureBias)
            {
                MergeTwoRgn(curid, idint, neiarr, rginfoarr, mergearr);
                cl = rginfoarr[curid].l;//��ǰ����LUVֵˢ�£�
                cu = rginfoarr[curid].u;
                cv = rginfoarr[curid].v;
                loopmerged = TRUE;
            }      
        }
    }
}
 
/**
 * @brief invoke AddBNeiToANei
 *
 * @param curid
 * @param 
 * @param neiarr
 * @param rginfoarr
 * @param mergearr
 */
void MyWatershed::MergeTwoRgn(INT curid, INT nearid
    ,CString* neiarr, MyRgnInfoWatershed* rginfoarr, INT* mergearr)
//��nearid�ϲ���curid��ȥ�����ºϲ��������Ϣ������¼�úϲ���
{
    //������Ϣ��nearid��Ӧ��ı����Ϊ�ѱ��ϲ���
    rginfoarr[nearid].isflag = TRUE;
    //���ºϲ����LUV��Ϣ��
    LONG ptincur = rginfoarr[curid].ptcount;
    LONG ptinnear = rginfoarr[nearid].ptcount;
    DOUBLE curpercent = (FLOAT)ptincur / (FLOAT)(ptincur+ptinnear);
    rginfoarr[curid].ptcount = ptincur + ptinnear;
    rginfoarr[curid].l = (FLOAT) ( curpercent * rginfoarr[curid].l
        + (1-curpercent) * rginfoarr[nearid].l );
    rginfoarr[curid].u = (FLOAT) ( curpercent * rginfoarr[curid].u
        + (1-curpercent) * rginfoarr[nearid].u );
    rginfoarr[curid].v = (FLOAT) ( curpercent * rginfoarr[curid].v
        + (1-curpercent) * rginfoarr[nearid].v );
    //��nearid������ӵ�curid��������ȥ��
    AddBNeiToANei(curid, nearid, neiarr, mergearr);
    //��¼�úϲ���
    mergearr[nearid] = curid;
}
 
void MyWatershed::AddBNeiToANei(INT curid, INT nearid, CString* neiarr, INT* mergearr)
//��nearid������ӵ�curid��������ȥ��
{
    //�ȴ�curid�������а�nearidɾȥ��
/*
    CString tempstr;
    tempstr.Format("%d ", nearid);
    INT temppos = neiarr[curid].Find(tempstr, 0);
    while (temppos>0 && neiarr[curid].GetAt(temppos-1)!=' ')
    {
        temppos = neiarr[curid].Find(tempstr, temppos+1);
    }
    if (temppos>=0)
    {
        //�����ڽ���Ϊ�ϲ��������������ԣ�
        neiarr[curid].Delete(temppos, tempstr.GetLength());
    }
*/
    //��nearid���������μӵ�curid��������ȥ��
    CString neistr = neiarr[nearid];
    CString curstr = neiarr[curid];
    //һ��˵������С��������Ӧ�ý��٣���ˣ�Ϊ����ߺϲ��ٶȣ���
    //curstr�ӵ�neistr��ȥ��Ȼ�󽫽������neiarr[curid];
    while ( curstr.GetLength()>0 )
    {
        INT pos = curstr.Find(_T(' '));    
        ASSERT(pos>=0);
        CString idstr = curstr.Left(pos);
        curstr.Delete(0, pos+1);
    INT idint = _wtoi(idstr);
        idint = FindMergedRgn(idint, mergearr);
        idstr += " ";
        if ( (idint == curid) || (idint == nearid) )
        {
            continue;//�������뱾�����ڣ�
        }else
        {
            if ( neistr.Find(idstr, 0) >= 0 )
            {
                continue;
            }else
            {
                neistr += idstr;//�ӵ�������ȥ;
            }
        }      
    }
    neiarr[curid] = neistr;
/*
    CString toaddneis = neiarr[nearid];
    while (toaddneis.GetLength()>0)
    {
        INT pos = toaddneis.Find(" ");     
        ASSERT(pos>=0);
        CString idstr = toaddneis.Left(pos);
        toaddneis.Delete(0, pos+1);
        INT idint = (INT) strtol(idstr, NULL, 10);
        idint = FindMergedRgn(idint, mergearr);
        idstr += " ";
        if ( (idint == curid) || (idint == nearid) )
        {
            continue;//�������뱾�����ڣ�
        }else
        {
            if ( neiarr[curid].Find(idstr, 0) >= 0 )
            {
                continue;
            }else
            {
                neiarr[curid] += idstr;//�ӵ�������ȥ;
            }
        }      
    }
*/
}
//### 
INT MyWatershed::FindNearestNei(INT curid, CString neistr, MyRgnInfoWatershed* rginfoarr, INT* mergearr)
//Ѱ��neistr����curid��ӽ����������ظ���id�ţ�
{
    INT outid = -1;
    DOUBLE mindis = 999999;
    FLOAT cl, cu, cv;
    cl = rginfoarr[curid].l;//��ǰ����LUVֵ��
    cu = rginfoarr[curid].u;
    cv = rginfoarr[curid].v;
 
    CString tempstr = neistr;//���ڱ������ڲ�����
    while (tempstr.GetLength()>0)
    {
        INT pos = tempstr.Find(_T(' '));
        ASSERT(pos>=0);
        CString idstr = tempstr.Left(pos);
        tempstr.Delete(0, pos+1);

	 INT idint = _wtoi(idstr);
        //�жϸ����Ƿ��ѱ��ϲ������ǣ���һֱ�ҵ�������ǰ�����ţ�
        idint = FindMergedRgn(idint, mergearr);
        if (idint==curid)
        {
            continue;//��������ѱ��ϲ�����ǰ����������
        }
        FLOAT tl, tu, tv;
        tl = rginfoarr[idint].l;//��ǰ�����������LUVֵ;
        tu = rginfoarr[idint].u;
        tv = rginfoarr[idint].v;
        DOUBLE tempdis = pow(tl-cl, 2)
            + pow(tu-cu, 2) + pow(tv-cv, 2);
        if (tempdis<mindis)
        {
            mindis = tempdis;//������Ͷ�Ӧ��������ID��
            outid = idint;
        }      
    }
 
    return outid;
}
//### 
void MyWatershed::AddNeiRgn(INT curid, INT neiid, CString* neiarr)
//����neiidΪcurid��������
{
    CString tempneis = neiarr[curid];//��ǰ����������
    CString toaddstr;
    toaddstr.Format(_T("%d "), neiid);
	//##
    INT temppos = tempneis.Find(toaddstr, 0);
    while (temppos>0 && neiarr[curid].GetAt(temppos-1)!=_T(' '))
    {
        temppos = neiarr[curid].Find(toaddstr, temppos+1);
    }
     
    if ( temppos<0 )
    {
        //��ǰ��������û��tempneis,�����
        neiarr[curid] += toaddstr;
    }
}
 
void MyWatershed::AddNeiOfCur(INT curid, INT left, INT right, INT up, INT down, INT* flag, CString* neiarr)
//ˢ�µ�ǰ���������������
{
    INT leftid, rightid, upid, downid;
    leftid = rightid = upid = downid = curid;
    if (left>=0)
    {
        leftid = flag[left];
        if (leftid!=curid)
        {
            //�ڵ�������һ��, ���������Ϣ��
            AddNeiRgn(curid, leftid, neiarr);
        }
    }
    if (right>0)
    {
        rightid = flag[right];
        if (rightid!=curid)
        {
            //�ڵ�������һ��, ���������Ϣ��
            AddNeiRgn(curid, rightid, neiarr);
        }
    }
    if (up>=0)
    {
        upid = flag[up];
        if (upid!=curid)
        {
            //�ڵ�������һ��, ���������Ϣ��
            AddNeiRgn(curid, upid, neiarr);
        }
    }
    if (down>0)
    {
        downid = flag[down];
        if (downid!=curid)
        {
            //�ڵ�������һ��, ���������Ϣ��
            AddNeiRgn(curid, downid, neiarr);
        }
    }
}

