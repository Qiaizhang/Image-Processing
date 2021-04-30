#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
//Shi - Tomasi
Mat srcImg, grayImg;
int thresholdValue = 50;
int maxCorners = 200;
RNG rng(12345);

void ShiTomasiDemo(int, void*)
{
    if (thresholdValue < 5)
    {
        thresholdValue = 5;  //���ٱ���5���ǵ�
    }
    vector<Point2f> corners;  //װ�ؽǵ�
    //������Ҫ�Ĳ���
    double quelityLevel = 0.02;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarris = false;
    double k = 0.04;
    Mat resultImg = srcImg.clone();  //���ƻҶ�ͼ���ͼ
    goodFeaturesToTrack(grayImg, corners, thresholdValue, quelityLevel, minDistance, Mat(), blockSize, useHarris, k);
    cout << "Number of Detected Corners��" << corners.size() << endl;

    for (int i = 0; i < corners.size(); i++)
    {
        cv::circle(resultImg, corners[i], 2, Scalar(0, 0, 255), 2);
        cout << corners[i] << endl;
    }
    imshow("ShiTomasi Detector", resultImg);
}

void test()
{
    srcImg = imread("D:\\Image\\12.jpg");
    if (srcImg.empty())
    {
        cout << "could not load image...\n" << endl;
    }
    namedWindow("Original image");
    namedWindow("ShiTomasi Detector");
    imshow("Original image", srcImg);

    cvtColor(srcImg, grayImg, COLOR_BGR2RGBA);
    createTrackbar("Threshold Value", "ShiTomasi Detector", &thresholdValue, maxCorners, ShiTomasiDemo);
    ShiTomasiDemo(0, 0);
}

int main()
{
    test();
    waitKey(0);
    return 0;
}