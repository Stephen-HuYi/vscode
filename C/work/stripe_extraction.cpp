//
//  stripe_extraction.cpp
//  opencvtest
//
//  Created by Jiawen Xue on 2020/4/10.
//  Copyright © 2020 Jiawen Xue. All rights reserved.
//


#include "configfiles.h"

void stripe_extraction(Mat src, Mat dst)
{
	dst = Mat::zeros(dst.rows,dst.cols,CV_8UC1);
	int h = src.rows;
    int w = src.cols;
	//Mat dst(h,w,CV_8UC1);
    double sigma = 0.6;
    
    Mat Prob1 = Mat::ones(h,w,CV_64F);
    Prob1 = -1000*Prob1;
    uchar T1 = 8;
    int win = 4;

    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            if(src.at<uchar>(i,j) >= T1)
            {
                double P_cur = 0.0;
                double B = (double)src.at<uchar>(i,j);
                double K = -B/(win+1)/(win+1);
                
                
                int i_range1 = (0>i-win)?0:(i-win);
                int i_range2 = (i+win<h-1)?(i+win):(h-1);
                
                for(int ii=i_range1;ii<=i_range2;ii++)
                {
                    double y_m = (double)src.at<uchar>(ii,j); //actual pixel value
                    double y_ideal = K*(i-ii)*(i-ii)+B;//ideal pixel value derived from y=Kx^2+B distribution
                    P_cur += abs(y_m-y_ideal);
                }
                
                Prob1.at<double>(i,j) = -P_cur/(8);
            }
        }
    }
    
    Mat Prob_matrix = Mat::zeros(h,w,CV_64F);
    
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            if(src.at<uchar>(i,j) >= T1)
            {
                
                int i_range1 = (0>i-8)?0:(i-8);
                int i_range2 = (i+8<h-1)?(i+8):(h-1);
                Mat prob_list = Mat::zeros(h,1,CV_64FC1);
                
                double Prob_max = -120.0;
                int Prob_loca = 1000;

                for(int k=i_range1;k<=i_range2;k++)
                {
                    int cur_range1 = (0>k-8)?0:(k-8);
                    int cur_range2 = (k+8<h-1)?(k+8):(h-1);
                    
                    //find the nearest stripe center point
                    if(j>0)
                    {
                        int min_dist = 1000;
                        for(int a=cur_range1;a<=cur_range2;a++)
                        {
                            if(dst.at<uchar>(a,j-1) == 255)
                            {
                                if(abs(k-a-1) < min_dist)
                                    min_dist = abs(k-a-1);
                            }
                        }
                        
                        if(min_dist<1000)//location is found
                            prob_list.at<double>(k,0) = -min_dist*min_dist/2/sigma/sigma +Prob1.at<double>(k,j);
                        else
                            prob_list.at<double>(k,0) = -8*8/2/sigma/sigma + Prob1.at<double>(k,j);
                    }
                    else
                        prob_list.at<double>(k,0) = -8*8/2/sigma/sigma + Prob1.at<double>(k,j);
                    
                    if(prob_list.at<double>(k,0)>=Prob_max)
                    {
                        Prob_max = prob_list.at<double>(k,0);
                        Prob_loca = k;
                    }
                }
                
                
                //    if(Prob_loca==i && Prob_max >= -120 && Prob1.at<double>(i,j) >= -15)
                if(Prob_loca==i && Prob1.at<double>(i,j) >= -15)
                {
                    //noise judgement
                    int i_noise_range1 = (0>i-8)?0:(i-8);
                    int i_noise_range2 = (i+8<h-1)?(i+8):(h-1);
                    double i_noise_max = -10000;
                    bool noise_found = false;
                    //Mat cur_col = dst.col(j).clone();
                    // cur_col = cur_col.rowRange(i_noise_range1,i_noise_range2).clone();
                    
                    vector<int> noise_loca_h;
                    for(int ni=i_noise_range1;ni<=i_noise_range2;ni++)
                    {
                        
                        uchar temp_cur_col = dst.at<uchar>(ni,j);
                        if(temp_cur_col==255)
                        {
                            double cur_noise_val = Prob_matrix.at<double>(ni,j);
                            if(cur_noise_val>i_noise_max)
                            {
                                i_noise_max = cur_noise_val;
                                noise_found = true;
                            }
                        }
                    }
        
                    if(!noise_found)
                    {                     
                        dst.at<uchar>(i,j) = 255;
                        Prob_matrix.at<double>(i,j) = Prob_max;
                    }
                    else if(i_noise_max < Prob_max)
                    {                     
                        for(int ni=i_noise_range1;ni<=i_noise_range2;ni++)
                            dst.at<uchar>(ni,j) = 0;
                        dst.at<uchar>(i,j) = 255;
                        Prob_matrix.at<double>(i,j) = Prob_max;
                    }
                }
            }
        }
    }
}

void cvStripeLocate_pre(Mat input_image,Mat map)
{
	map =Mat::zeros(map.rows,map.cols,CV_8UC1);
	IplImage* src = &IplImage(input_image);

	//粗定位,生成条纹点阵
	int pixel[9];
	for(int i=0;i<src->height;i++)
	{	
		for(int j=0;j<src->width;j++)
		{
			map.at<uchar>(i,j)=false;
			if(j>=5&&j<=src->width-5&&i>=5&&i<src->height-5)
			{
				int ia=i;//+pi->startY;
				int ja=j;//+pi->startX;
				pixel[0]=((uchar *)(src->imageData + src->widthStep * (ia-1)))[ja];   // ((uchar *)(src->imageData + src->widthStep * ia))[ja-1];//cvGet2D(src,ia,ja-1);
				pixel[1]=((uchar *)(src->imageData + src->widthStep * ia))[ja];
				pixel[2]=((uchar *)(src->imageData + src->widthStep * (ia+1)))[ja];   // (src->imageData + src->widthStep * ia))[ja+1];
				pixel[3]=((uchar *)(src->imageData + src->widthStep * (ia-2)))[ja];   // (src->imageData + src->widthStep * ia))[ja-2];
				pixel[4]=((uchar *)(src->imageData + src->widthStep * (ia+2)))[ja];   // (src->imageData + src->widthStep * ia))[ja+2];
				pixel[5]=((uchar *)(src->imageData + src->widthStep * (ia+3)))[ja];   // (src->imageData + src->widthStep * ia))[ja+3];
				pixel[6]=((uchar *)(src->imageData + src->widthStep * (ia-3)))[ja];   // (src->imageData + src->widthStep * ia))[ja-3];
				pixel[7]=((uchar *)(src->imageData + src->widthStep * (ia+4)))[ja];   // (src->imageData + src->widthStep * ia))[ja+4];
				pixel[8]=((uchar *)(src->imageData + src->widthStep * (ia-4)))[ja];   // (src->imageData + src->widthStep * ia))[ja-4];

				if(pixel[1]>/*pi->strength*0.5*/10&&pixel[1]>pixel[0]&&pixel[1]>pixel[3]&&pixel[1]>=pixel[2]&&pixel[1]>=pixel[4]
				&&pixel[1]>=pixel[5]&&pixel[1]>pixel[6]&&pixel[1]>=pixel[7]&&pixel[1]>pixel[8]
				&&pixel[0]>=pixel[3]&&pixel[3]>=pixel[6]&&pixel[2]>=pixel[5]&&pixel[5]>=pixel[7]
				&& (pixel[1] - pixel[8] > 3)
                && (pixel[1] - pixel[6] > 1)
				)
				{
					int max_value = -1;
					int min_value = 256;
					for (int ii = 0; ii<9; ii+=1)
					{
						if(pixel[ii] > max_value)
							max_value = pixel[ii];
						if(pixel[ii] < min_value)
							min_value = pixel[ii];
					}
					if (max_value-min_value>8)
						map.at<uchar>(i,j) = 255;
				}
			}
		}
	}
}