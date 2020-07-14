/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-06-27 20:01:14
 * @LastEditors: HuYi
 * @LastEditTime: 2020-06-27 20:01:44
 */
int otsu(IplImage *image)
{
    assert(NULL != image);
    int width = image->width;
    int height = image->height;
    int x = 0, y = 0;
    int pixelCount[256];
    float pixelPro[256];
    int i, j, pixelSum = width * height, threshold = 0;

    uchar *data = (uchar *)image->imageData;

    //初始化
    for (i = 0; i < 256; i++)
    {
        pixelCount[i] = 0;
        pixelPro[i] = 0;
    }

    //统计灰度级中每个像素在整幅图像中的个数
    for (i = y; i < height; i++)
    {
        for (j = x; j < width; j++)
        {
            pixelCount[data[i * image->widthStep + j]]++;
        }
    }

    //计算每个像素在整幅图像中的比例
    for (i = 0; i < 256; i++)
    {
        pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
    }

    //经典ostu算法,得到前景和背景的分割
    //遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
    float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
    for (i = 0; i < 256; i++)
    {
        w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;

        for (j = 0; j < 256; j++)
        {
            if (j <= i) //背景部分
            {
                //以i为阈值分类，第一类总的概率
                w0 += pixelPro[j];
                u0tmp += j * pixelPro[j];
            }
            else //前景部分
            {
                //以i为阈值分类，第二类总的概率
                w1 += pixelPro[j];
                u1tmp += j * pixelPro[j];
            }
        }

        u0 = u0tmp / w0;   //第一类的平均灰度
        u1 = u1tmp / w1;   //第二类的平均灰度
        u = u0tmp + u1tmp; //整幅图像的平均灰度
                           //计算类间方差
        deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);
        //找出最大类间方差以及对应的阈值
        if (deltaTmp > deltaMax)
        {
            deltaMax = deltaTmp;
            threshold = i;
        }
    }
    //返回最佳阈值;
    return threshold;
}