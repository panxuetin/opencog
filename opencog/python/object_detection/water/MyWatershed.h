#pragma once
class MyRgnInfoWatershed
{
public:
	LONG ptcount;
	BOOL isflag;
	FLOAT l;
	FLOAT u;
	FLOAT v;
	
	
};
class MyLUV
{
public:
	FLOAT l;
	FLOAT u;
	FLOAT v;
	
};
class MyImageGraPtWatershed
{
public:
	INT gradient;
	INT x;
	INT y;
};

#include <highgui.h>
#include <WinDef.h>
#include <afx.h>
class MyWatershed
{
public:
	MyWatershed(void);
	~MyWatershed(void);


	void WatershedSegmentVincent(IplImage* img);
void GetGra(IplImage* image, INT* deltar);

void Flood(MyImageGraPtWatershed* imiarr, INT* graddarr,
                         INT minh, INT maxh,
                         INT* flagarr, INT& outrgnumber,
                         INT width,INT height);

void MergeRgs(MyRgnInfoWatershed* rginfoarr,
                            INT rgnumber, INT* flag, INT width,
                            INT height, INT* outmerge, INT& rgnum);
void MergeNearest(INT curid, MyRgnInfoWatershed* rginfoarr, CString* neiarr, INT* mergearr);

void MergeTwoRgn(INT curid, INT nearid
    , CString* neiarr, MyRgnInfoWatershed* rginfoarr, INT* mergearr);

void AddBNeiToANei(INT curid, INT nearid, CString* neiarr, INT* mergearr);

INT FindNearestNei(INT curid, CString neistr, MyRgnInfoWatershed* rginfoarr, INT* mergearr);

INT FindMergedRgnMaxbias(INT idint, INT* mergearr, INT bias);

INT FindMergedRgn(INT idint, INT* mergearr);

void AddNeiRgn(INT curid, INT neiid, CString* neiarr);

void AddNeiOfCur(INT curid, INT left, INT right, INT up, INT down, INT* flag, CString* neiarr);


};

