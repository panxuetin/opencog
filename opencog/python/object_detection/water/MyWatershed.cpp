#include "StdAfx.h"
#include "MyWatershed.h"
#define NearMeasureBias 200.0//判定区域颜色相似的阈值；
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
void GetNeiInt(INT x, INT y, INT pos, INT width, INT height, INT &left, INT &right, INT &up, INT &down)//找pos的四个邻域；
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
//分水岭分割：img为彩色
{  
    int x,y,i; //循环
 
    IplImage* pSrc = NULL;
    pSrc = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
    cvCopy(img,pSrc);
    int width = pSrc->width;         //图像宽度
    int height = pSrc->height;           //图像高度
    int depth = pSrc->depth;         //图像位深(IPL_DEPTH_8U...)
    int channels = pSrc->nChannels;      //图像通道数(1、2、3、4)
    int imgSize = pSrc->imageSize;       //图像大小 imageSize = height*widthStep
    int step = pSrc->widthStep/sizeof(uchar);   
    uchar* data    = (uchar *)pSrc->imageData;


    int imageLen = width*height;   
/*
    cvNamedWindow("src");f##
    cvShowImage("src",pSrc);
    */
     
    /*
    //===================通过Sobel求取梯度图像====================//
    //  彩色转灰度
    IplImage* gray  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    cvNamedWindow("gray");
    cvShowImage("gray",gray);
     
    //  Sobel
    IplImage* sobel = cvCreateImage(cvGetSize(pSrc),IPL_DEPTH_16S,1); //IPL_DEPTH_16S: 以Sobel方式求完导数后会有负值，还有会大于255的值
    cvSobel(gray,sobel,1,1,3);
    IplImage *sobel8u=cvCreateImage(cvGetSize(sobel),IPL_DEPTH_8U,1);
    cvConvertScaleAbs(sobel,sobel8u,1,0); //转为8位无符号
     
    //  临时显示sobel
    cvNamedWindow("sobel8u");
    cvShowImage("sobel8u",sobel8u); */
     
     
    //===================① 获取原彩色图像的梯度值(先转灰度)===================//
    int* deltar = new INT[imageLen]; //梯度模数组
    //FLOAT* deltasita = new FLOAT[imgSize];//梯度角度数组；
    //  彩色转灰度
    IplImage* gray  = cvCreateImage(cvGetSize(pSrc),pSrc->depth,1);
    cvCvtColor(pSrc,gray,CV_BGR2GRAY); 
    //  求梯度
    GetGra(gray,deltar);
    //GetGra2(sobel8u,deltar);
     
 
    //===================② 梯度值排序===================//
    //以下统计各梯度频率
    MyImageGraPtWatershed*  graposarr = new MyImageGraPtWatershed[imageLen];
    INT*  gradientfre = new INT[256];//图像中各点梯度值频率；
    INT*  gradientadd = new INT[257];//各梯度起终位置；
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
            gradientfre[tempi] ++;//灰度值频率；
        }
    }
     
    //统计各梯度的累加概率；
    INT added = 0;
    gradientadd[0] = 0;//第一个起始位置为0；
    for (INT ii=1; ii<256; ii++)
    {
        added += gradientfre[ii-1];
        gradientadd[ii] = added;
    }
    gradientadd[256] = imageLen;//最后位置；
     
    memset( gradientfre, 0, 256*sizeof(INT)); //清零，下面用作某梯度内的指针；
     
    //自左上至右下sorting....
    for (y=0; y<height; y++)
    {
        xstart = y*width;
        for ( x=0; x<width; x++)
        {
            deltapos = xstart + x;
            INT tempi = (INT)(deltar[deltapos]);//当前点的梯度值，由于前面的步骤，最大只能为255；
            //根据梯度值决定在排序数组中的位置；
            INT tempos = gradientadd[tempi] + gradientfre[tempi];
            gradientfre[tempi] ++;//梯度内指针后移；
            graposarr[tempos].gradient = tempi; //根据当前点的梯度将该点信息放后排序后数组中的合适位置中去；    
            graposarr[tempos].x = x;
            graposarr[tempos].y = y;
        }
    }
 
 
    //===================③ 泛洪得每个像素所属区域标记flag=================//
    INT rgnumber = 0;                   //分割后的区域数
    INT*   flag = new INT[imageLen];        //各点标识数组
    Flood(graposarr,gradientadd,0,255,flag,rgnumber,width,height); //!rgnumber = the number of area 
	//printf("%d******************%d",rgnumber,imageLen);

	//for(int i=0;i<imageLen;i++)
	{
		//printf("%d*",flag[i]);
		//if(!(i%90))
		//	printf("\n");
	}
     
     
    //===================④ 由flag标记求各个区域的LUV均值=================//
    //以下准备计算各个区域的LUV均值；
    //分割后各个区的一些统计信息,第一个元素不用，图像中各点所属区域的信息存放在flag数组中
    MyRgnInfoWatershed*  rginfoarr = new MyRgnInfoWatershed[rgnumber+1];   
    //清空该数组
    for (i=0; i<rgnumber+1; i++) //LUV
    {
        rginfoarr[i].isflag = FALSE;
        rginfoarr[i].ptcount = 0;
        rginfoarr[i].l = 0;
        rginfoarr[i].u = 0;
        rginfoarr[i].v = 0;
    }
     
    //以下保存LUV数据；
	

	
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
            INT rgid = flag[pos];//当前位置点所属区域在区统计信息数组中的位置
            //以下将该点的信息加到其所属区信息中去
            rginfoarr[rgid].ptcount ++;
            rginfoarr[rgid].l += luvData[pos].l;
            rginfoarr[rgid].u += luvData[pos].u;
            rginfoarr[rgid].v += luvData[pos].v;
        }
    }
 
    //求出各个区的LUV均值；
    for (i=0; i<=rgnumber; i++)
    {
        rginfoarr[i].l = (FLOAT) ( rginfoarr[i].l / rginfoarr[i].ptcount );
        rginfoarr[i].u = (FLOAT) ( rginfoarr[i].u / rginfoarr[i].ptcount );
        rginfoarr[i].v = (FLOAT) ( rginfoarr[i].v / rginfoarr[i].ptcount );
    }
 
    //===================⑤ 区域融合 ===================//
    //根据各区luv均值（rginfoarr）和各区之间邻接关系（用flag计算）进行区域融合
    INT* mergearr = new INT[rgnumber+1];
    memset( mergearr, -1, sizeof(INT)*(rgnumber+1) );
    INT mergednum = 0;
    //entrance
    MergeRgs(rginfoarr, rgnumber, flag, width, height, mergearr, mergednum);
    //MergeRgs(rginfoarr, rgnumber, flag, width, height);
     
 
    //===================⑥ 确定融合后各个像素所属区域===================//
    //确定合并后各像素点所属区域；
    for (y=0; y<(height); y++)
    {
        xstart = y*width;
        for ( x=0; x<(width); x++)
        {
            INT pos = xstart + x;
            INT rgid = flag[pos];//该点所属区域；
            flag[pos] = FindMergedRgn(rgid, mergearr);
        }
    }
    delete [] mergearr;
    mergearr = NULL;
 
    //===================⑧ 绘制区域轮廓，保存结果===================//
    //=================== ===================//
    //以下根据标识数组查找边界点（不管四边点以减小复杂度）
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
    //  在源图像上绘制轮廓
    CvMemStorage* storage= cvCreateMemStorage(0);
    CvSeq* contour= 0;//可动态增长元素序列
    IplImage* dstContourGray= cvCreateImage(cvGetSize(dstContour),8,1);
    cvCvtColor(dstContour,dstContourGray,CV_BGR2GRAY);
    cvNamedWindow("dstContourGray",1);
    cvShowImage("dstContourGray",dstContourGray);
 
    cvFindContours( //函数cvFindContours从二值图像中检索轮廓，并返回检测到的轮廓的个数
        dstContourGray,//二值化图像
        storage,    //轮廓存储容器
        &contour,   //指向第一个轮廓输出的指针
        sizeof(CvContour),  //序列头大小
        CV_RETR_CCOMP,      //内外轮廓都检测; CV_RETR_EXTERNAL：只检索最外面的轮廓
        CV_CHAIN_APPROX_SIMPLE//压缩水平垂直和对角分割，即函数只保留末端的象素点 CV_CHAIN_CODE//CV_CHAIN_APPROX_NONE
        );
     
    IplImage* dst = cvCreateImage(cvGetSize(pSrc),IPL_DEPTH_8U,3);
    cvCopy(pSrc,dst);
    for (;contour!= 0;contour= contour->h_next)
    {
        CvScalar color= CV_RGB(rand()&255,rand()&255,rand()&255);
        cvDrawContours( //在图像中绘制轮廓
            dst,
            contour,//指向初始轮廓的指针
            color,  //外层轮廓的颜色
            color,  //内层轮廓的颜色
            -1,     //绘制轮廓的最大等级（
                        //0：当前；
                        //1：相同级别下的轮廓；
                        //2：绘制同级与下一级轮廓；
                        //负数：不会绘制当前轮廓之后的轮廓，但会绘制子轮廓，到（abs(该参数)-1）为止）
            1,      //轮廓线条粗细
            8       //线条的类型
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
 
    //  释放资源
    delete [] gradientadd; gradientadd = NULL;//大小256
    delete [] gradientfre; gradientfre = NULL;//大小256
    delete [] deltar;    deltar    = NULL;//大小imageLen
    delete [] graposarr; graposarr = NULL;//大小imageLen
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
//通过sobel获取灰度图像image的梯度值存数组deltar
{
    //   定义
    IplImage* pSrc = NULL;
    pSrc = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
    cvCopy(image,pSrc);
    int width = pSrc->width;
    int height = pSrc->height;
 
/*  测试参数
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
 
    //暂不计算边缘点
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
     
    //边缘赋为其内侧点的值
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
        INT inner = x + width;//下一行；
        deltar[pos1] = deltar[inner];
        INT y2 = height-1;
        INT pos2 = y2*width + x;
        inner = pos2 - width;//上一行；
        deltar[pos2] = deltar[inner];
    }
     
    delete [] gra;
    gra=NULL;
 
}
 
 
//////////////////////////////////////////////////////////////////////////
// Luc Vincent and Pierre Soille的分水岭分割flood步骤的实现代码，
// 修改自相应伪代码, 伪代码来自作者论文《Watersheds in Digital Spaces:
// An Efficient Algorithm Based on Immersion Simulations》
// IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE.
// VOL.13, NO.6, JUNE 1991;
// by dzj, 2004.06.28
// MyImageGraPt* imiarr - 输入的排序后数组
// INT* graddarr -------- 输入的各梯度数组，由此直接存取各H像素点
// INT minh，INT maxh --- 最小最大梯度
// INT* flagarr --------- 输出标记数组
// 注意：目前不设分水岭标记，只设每个像素所属区域；
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
    INT imagelen = imageWidth * imageHeight;//图像中像素的个数
    for ( i=0; i<imagelen; i++)
    {
        flagarr[i] = INIT;
    }
    //memset(flagarr, INIT, sizeof(INT)*imagelen);
    INT* imd = new INT[imagelen]; //距离数组，直接存取；
    for (i=0; i<imagelen; i++)
    {
        imd[i] = 0;
    }
    //memset(imd, 0, sizeof(INT)*imagelen);
    std::queue <int> myqueue;
    INT curlabel = 0;//各盆地标记；
     
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
            //以下检查该点邻域是否已标记属于某区或分水岭，若是，则将该点加入fifo;
            INT left = ipos - 1;
            if (x-1>=0)
            {
                if (flagarr[left]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//点位置压入fifo;
                    continue;
                }              
            }
            INT right = ipos + 1;
            if (x+1<imageWidth)
            {
                if (flagarr[right]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//点位置压入fifo;
                    continue;
                }
            }
            INT up = ipos - imageWidth;
            if (y-1>=0)
            {
                if (flagarr[up]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//点位置压入fifo;
                    continue;
                }              
            }
            INT down = ipos + imageWidth;
            if (y+1<imageHeight)
            {
                if (flagarr[down]>=0)
                {
                    imd[ipos] = 1;
                    myqueue.push(ipos);//点位置压入fifo;
                    continue;
                }          
            }
        }
         
        //以下根据先进先出队列扩展现有盆地；
        INT curdist = 1; myqueue.push(-99);//特殊标记；
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
             
            //以下找p的邻域；
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
                        //ppei属于某区域（不是分水岭）；
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //将其设为邻点所属区域；
                            flagarr[p] = flagarr[left];
                        }else if (flagarr[p]!=flagarr[left])
                        {
                            //原来赋的区与现在赋的区不同，设为分水岭；
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppei为分岭；
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[left]==MASK) && (imd[left]==0) )
                    //ppei中已MASK的点，但尚未标记（即不属某区也不是分水岭）;
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
                        //ppei属于某区域（不是分水岭）；
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //将其设为邻点所属区域；
                            flagarr[p] = flagarr[right];
                        }else if (flagarr[p]!=flagarr[right])
                        {
                            //原来赋的区与现在赋的区不同，设为分水岭；
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppei为分岭；
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[right]==MASK) && (imd[right]==0) )
                    //ppei中已MASK的点，但尚未标记（即不属某区也不是分水岭）;
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
                        //ppei属于某区域（不是分水岭）；
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //将其设为邻点所属区域；
                            flagarr[p] = flagarr[up];
                        }else if (flagarr[p]!=flagarr[up])
                        {
                            //原来赋的区与现在赋的区不同，设为分水岭；
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppei为分岭；
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[up]==MASK) && (imd[up]==0) )
                    //ppei中已MASK的点，但尚未标记（即不属某区也不是分水岭）;
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
                        //ppei属于某区域（不是分水岭）；
                        if ( (flagarr[p]==MASK)
                            || (flagarr[p]==WATERSHED) )
                        {
                            //将其设为邻点所属区域；
                            flagarr[p] = flagarr[down];
                        }else if (flagarr[p]!=flagarr[down])
                        {
                            //原来赋的区与现在赋的区不同，设为分水岭；
                            //flagarr[p] = WATERSHED;
                        }
                    }else if (flagarr[p]==MASK)//ppei为分岭；
                    {
                        flagarr[p] = WATERSHED;
                    }
                }else if ( (flagarr[down]==MASK) && (imd[down]==0) )
                    //ppei中已MASK的点，但尚未标记（既不属某区也不是分水岭）;
                {
                    imd[down] = curdist + 1; myqueue.push(down);
                }  
            }
             
        }//以上现有盆地的扩展；
         
        //以下处理新发现的盆地；
        for (INT ini=stpos; ini<edpos; ini++)
        {
            INT x = imiarr[ini].x;
            INT y = imiarr[ini].y;
            INT ipos = y*imageWidth + x;
            imd[ipos] = 0;//重置所有距离
            if (flagarr[ipos]==MASK)
            {
                //经过前述扩展后该点仍为MASK，则该点必为新盆地的一个起始点;
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
                        myqueue.push(ppeileft);//点位置压入fifo;
                        flagarr[ppeileft] = curlabel;
                    }
                    INT ppeiright = ppei + 1;
                    if ( (ppeix+1<imageWidth) && (flagarr[ppeiright]==MASK) )
                    {
                        myqueue.push(ppeiright);//点位置压入fifo;
                        flagarr[ppeiright] = curlabel;
                    }
                    INT ppeiup = ppei - imageWidth;
                    if ( (ppeiy-1>=0) && (flagarr[ppeiup]==MASK) )
                    {
                        myqueue.push(ppeiup);//点位置压入fifo;
                        flagarr[ppeiup] = curlabel;
                    }
                    INT ppeidown = ppei + imageWidth;
                    if ( (ppeiy+1<imageHeight) && (flagarr[ppeidown]==MASK) )
                    {
                        myqueue.push(ppeidown);//点位置压入fifo;
                        flagarr[ppeidown] = curlabel;
                    }                  
                }              
            }
        }//以上处理新发现的盆地；
         
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
              
               int curdist=1;//距离标志,个人感觉标志着是否被处理了
               myqueue.push(-99);//特殊标记
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
               //搜索p的临点
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
                    
                      
                       //搜索临点的for
                       for (int jj=stpos;jj<edpos;jj++)
                       {
                       x=imiarr[jj].x;
                       y=imiarr[jj].y;
                       INT ipos = y*width + x;
                       imd[ipos] = 0;//重置所有距离
                       if (flagarr[ipos]==MASK)
                       {
                       //经过前述扩展后该点仍为MASK，则该点必为新盆地的一个起始点;
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
                         myqueue.push(ppeileft);//点位置压入fifo;
                         flagarr[ppeileft] = curlabe;
                         }
                         INT ppeiright = ppei + 1;
                         if ( (ppeix+1<width) && (flagarr[ppeiright]==MASK) )
                         {
                         myqueue.push(ppeiright);//点位置压入fifo;
                         flagarr[ppeiright] = curlabe;
                         }
                         INT ppeiup = ppei - width;
                         if ( (ppeiy-1>=0) && (flagarr[ppeiup]==MASK) )
                         {
                         myqueue.push(ppeiup);//点位置压入fifo;
                         flagarr[ppeiup] = curlabe;
                         }
                         INT ppeidown = ppei + width;
                         if ( (ppeiy+1<height) && (flagarr[ppeidown]==MASK) )
                         {
                         myqueue.push(ppeidown);//点位置压入fifo;
                         flagarr[ppeidown] = curlabe;
                         }                 
                         }             
                         }
                         }//搜索临点的for
                          
                            
                             }//最后的一个for
                              
                               outrgnumber=curlabe;
                               delete [] imd;
                               imd=NULL;
*/
}
 

//********************************************************************************* 
/**
 * @brief entrance point
 *
 * @param rginfoarr 各个区的信息
 * @param rgnumber
 * @param flag
 * @param width
 * @param height
 * @param outmerge
 * @param rgnum
 */
void MyWatershed::MergeRgs(MyRgnInfoWatershed* rginfoarr, INT rgnumber, INT* flag, INT width, INT height, INT* outmerge, INT& rgnum)
//合并相似区域；
{
    //////////////////////////////////////////////////////////////////////////
    //1、建立各区的邻域数组；
    //2、依次扫描各区域，寻找极小区域；
    //3、对每个极小区（A），在相邻区中找到最相似者；
    //4、与相似区（B）合并（各种信息刷新），在极小区（A）的邻域中
    //   删除相似区（B），在邻域数组中删除相似区（B）对应的项，将
    //   相似区（B）的相邻区s加到极小区（A）的邻域中去；
    //5、记录合并信息，设一数组专门存放该信息，该数组的第A个元素值设为B；
    //6、判断是否仍为极小区，若是则返回3；
    //7、是否所有区域都已处理完毕，若非则返回2；
    //
    //   由于各区的相邻区不会太多，因此采用邻接数组作为存储结构；
    //////////////////////////////////////////////////////////////////////////
    CString* neiarr = new CString[rgnumber+1];//第一个不用；
    INT* mergearr = outmerge;//记录合并情况数组；
 
    //建立邻域数组；
    for (INT y=0; y<height; y++)
    {
        INT lstart = y * width;
        for (INT x=0; x<width; x++)
        {
            INT pos = lstart + x;
            INT left=-1, right=-1, up=-1, down=-1;
           //## pMyMath.
			GetNeiInt(x, y, pos, width, height, left, right, up, down);//找pos的四个邻域；
            //确定并刷新邻域区信息；
            INT curid = flag[pos];
            AddNeiOfCur(curid, left, right, up, down, flag, neiarr);
        }
    }//建立邻域数组；
     
    //区域信息数组中的有效信息从1开始，第i个位置存放第i个区域的信息；
    for (INT rgi=1; rgi<=rgnumber; rgi++)
    {
        //扫描所有区域，找极小区；
        LONG allpoints = width * height;
        LONG nmin = (LONG) (allpoints / 400);
        INT curid = rgi;
 
        //rginfoarr[rgi].isflag初始为FALSE，在被合并到其它区后改为TRUE；
        while ( ( (rginfoarr[rgi].ptcount)<nmin )
            && !rginfoarr[rgi].isflag )
        {
            //该区为极小区，遍历所有相邻区，找最接近者；
            CString neistr = neiarr[curid];
            INT nearid = FindNearestNei(curid, neistr, rginfoarr, mergearr);
            //合并curid与nearid；
            MergeTwoRgn(curid, nearid, neiarr, rginfoarr, mergearr);           
        }
    }
 
    //以下再合并相似区域，（无论大小）,如果不需要，直接将整个循环注释掉就行了；
    INT countjjj = 0;
    //区域信息数组中的有效信息从1开始，第i个位置存放第i个区域的信息；
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
//大阈值终止查找合并区，用于coarse watershed,
//调用者必须保证idint有效，即：mergearr[idint]>0；
//以及mergearr有效，即：mergearr[idint]<idint;
{
    INT outid = idint;
    while ( mergearr[outid]<bias )
    {
        outid = mergearr[outid];
    }
    return mergearr[outid];
}
 
INT MyWatershed::FindMergedRgn(INT idint, INT* mergearr)
//找到idint最终所合并到的区号；
{
    INT outid = idint;
    while ( mergearr[outid] > 0 )
    {
        outid = mergearr[outid];
    }
    return outid;
}
 

/**
 * @brief 合并相似区域；
 *
 * @param curid
 * @param rginfoarr
 * @param neiarr
 * @param mergearr
 */
void MyWatershed::MergeNearest(INT curid, MyRgnInfoWatershed* rginfoarr, CString* neiarr, INT* mergearr)
{
    //依次处理各个邻域，若相似，则合并；
    //CString neistr = neiarr[curid];
    FLOAT cl, cu, cv;
    cl = rginfoarr[curid].l;//当前区的LUV值；
    cu = rginfoarr[curid].u;
    cv = rginfoarr[curid].v;
    BOOL loopmerged = TRUE;//一次循环中是否有合并操作发生，若无，则退出循环；
 
    while (loopmerged)
    {
        loopmerged = FALSE;
        CString tempstr = neiarr[curid];//用于本函数内部处理；
        while (tempstr.GetLength()>0)
        {
            INT pos = tempstr.Find(_T(' '));
            ASSERT(pos>=0);
            CString idstr = tempstr.Left(pos);
            tempstr.Delete(0, pos+1);
      //       idstr.string
          INT idint = _wtoi(idstr);
            //判断该区是否已被合并，若是，则一直找到该区当前的区号；
            idint = FindMergedRgn(idint, mergearr);
            if (idint==curid)
            {
                continue;//这个邻区已被合并到当前区，跳过；
            }
            FLOAT tl, tu, tv;
            tl = rginfoarr[idint].l;//当前处理的邻区的LUV值;
            tu = rginfoarr[idint].u;
            tv = rginfoarr[idint].v;
            DOUBLE tempdis = pow(tl-cl, 2)
                + pow(tu-cu, 2) + pow(tv-cv, 2);
            if (tempdis<NearMeasureBias)
            {
                MergeTwoRgn(curid, idint, neiarr, rginfoarr, mergearr);
                cl = rginfoarr[curid].l;//当前区的LUV值刷新；
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
//将nearid合并到curid中去，更新合并后的区信息，并记录该合并；
{
    //将区信息中nearid对应项的标记设为已被合并；
    rginfoarr[nearid].isflag = TRUE;
    //更新合并后的LUV信息；
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
    //将nearid的邻域加到curid的邻域中去；
    AddBNeiToANei(curid, nearid, neiarr, mergearr);
    //记录该合并；
    mergearr[nearid] = curid;
}
 
void MyWatershed::AddBNeiToANei(INT curid, INT nearid, CString* neiarr, INT* mergearr)
//将nearid的邻域加到curid的邻域中去；
{
    //先从curid的邻区中把nearid删去；
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
        //否则邻近区为合并过来的区，忽略；
        neiarr[curid].Delete(temppos, tempstr.GetLength());
    }
*/
    //将nearid的邻区依次加到curid的邻区中去；
    CString neistr = neiarr[nearid];
    CString curstr = neiarr[curid];
    //一般说来，极小区的邻域应该较少，因此，为着提高合并速度，将
    //curstr加到neistr中去，然后将结果赋给neiarr[curid];
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
            continue;//本区不与本区相邻；
        }else
        {
            if ( neistr.Find(idstr, 0) >= 0 )
            {
                continue;
            }else
            {
                neistr += idstr;//加到邻区中去;
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
            continue;//本区不与本区相邻；
        }else
        {
            if ( neiarr[curid].Find(idstr, 0) >= 0 )
            {
                continue;
            }else
            {
                neiarr[curid] += idstr;//加到邻区中去;
            }
        }      
    }
*/
}
//### 
INT MyWatershed::FindNearestNei(INT curid, CString neistr, MyRgnInfoWatershed* rginfoarr, INT* mergearr)
//寻找neistr中与curid最接近的区，返回该区id号；
{
    INT outid = -1;
    DOUBLE mindis = 999999;
    FLOAT cl, cu, cv;
    cl = rginfoarr[curid].l;//当前区的LUV值；
    cu = rginfoarr[curid].u;
    cv = rginfoarr[curid].v;
 
    CString tempstr = neistr;//用于本函数内部处理；
    while (tempstr.GetLength()>0)
    {
        INT pos = tempstr.Find(_T(' '));
        ASSERT(pos>=0);
        CString idstr = tempstr.Left(pos);
        tempstr.Delete(0, pos+1);

	 INT idint = _wtoi(idstr);
        //判断该区是否已被合并，若是，则一直找到该区当前的区号；
        idint = FindMergedRgn(idint, mergearr);
        if (idint==curid)
        {
            continue;//这个邻区已被合并到当前区，跳过；
        }
        FLOAT tl, tu, tv;
        tl = rginfoarr[idint].l;//当前处理的邻区的LUV值;
        tu = rginfoarr[idint].u;
        tv = rginfoarr[idint].v;
        DOUBLE tempdis = pow(tl-cl, 2)
            + pow(tu-cu, 2) + pow(tv-cv, 2);
        if (tempdis<mindis)
        {
            mindis = tempdis;//最大距离和对应的相邻区ID；
            outid = idint;
        }      
    }
 
    return outid;
}
//### 
void MyWatershed::AddNeiRgn(INT curid, INT neiid, CString* neiarr)
//增加neiid为curid的相邻区
{
    CString tempneis = neiarr[curid];//当前的相邻区；
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
        //当前相邻区中没有tempneis,则加入
        neiarr[curid] += toaddstr;
    }
}
 
void MyWatershed::AddNeiOfCur(INT curid, INT left, INT right, INT up, INT down, INT* flag, CString* neiarr)
//刷新当前点的所有相邻区；
{
    INT leftid, rightid, upid, downid;
    leftid = rightid = upid = downid = curid;
    if (left>=0)
    {
        leftid = flag[left];
        if (leftid!=curid)
        {
            //邻点属于另一区, 加邻域点信息；
            AddNeiRgn(curid, leftid, neiarr);
        }
    }
    if (right>0)
    {
        rightid = flag[right];
        if (rightid!=curid)
        {
            //邻点属于另一区, 加邻域点信息；
            AddNeiRgn(curid, rightid, neiarr);
        }
    }
    if (up>=0)
    {
        upid = flag[up];
        if (upid!=curid)
        {
            //邻点属于另一区, 加邻域点信息；
            AddNeiRgn(curid, upid, neiarr);
        }
    }
    if (down>0)
    {
        downid = flag[down];
        if (downid!=curid)
        {
            //邻点属于另一区, 加邻域点信息；
            AddNeiRgn(curid, downid, neiarr);
        }
    }
}

