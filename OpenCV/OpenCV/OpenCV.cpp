// OpenCV.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat image;
Mat imageGray;
int Thresh = 200;
const int MaxThresh = 255;

void Trackbar(int, void*) {
	Mat dst, dst8u, dstshow, imageSource;
	dst = Mat::zeros(image.size(), CV_32FC1);
	imageSource = image.clone();
	cv::cornerHarris(imageGray, dst, 3, 3, 0.1, BORDER_DEFAULT);
	cv::normalize(dst, dst8u, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(dst8u, dstshow);
	imshow("dst", dstshow);

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (dstshow.at<uchar>(i, j) > Thresh)
				cv::circle(imageSource, Point(j, i), 2, Scalar(0, 0, 255), 2);
		}
	}

	imshow("Corner Detected", imageSource);
}

//int main() {
//	image = imread("D:\\Image\\11-cs.jpg");
//
//	cv::cvtColor(image, imageGray, COLOR_BGR2GRAY);
//	cv::namedWindow("Corner Detected");
//	cv::createTrackbar("threshold: ", "Corner Detected", &Thresh, MaxThresh, Trackbar);
//	cv::imshow("corner ", image);
//	Trackbar(0, 0);
//	waitKey();
//
//
//
//	return 0;
//}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
