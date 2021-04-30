// 滤镜.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<cmath>
#include<opencv2/opencv.hpp>

// black and white filter:黑白滤镜
cv::Mat bwFilter(cv::Mat img)
{
	cv::Mat result(img.size(), CV_8UC1);
	cv::cvtColor(img, result, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	const int Thresh = 128;

	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			if (result.at<uchar>(i, j) > Thresh)
			{
				result.at<uchar>(i, j) = 255;
			}
			else
			{
				result.at<uchar>(i, j) = 0;
			}
		}
	}
	return result;
}

// reversal-filter.cpp ：反向滤镜
cv::Mat rFilter(cv::Mat img) 
{
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			for (size_t k = 0; k < 3; k++) {
				result.at<cv::Vec3b>(i, j)[k] = 255 - img.at<cv::Vec3b>(i, j)[k];
			}

		}
	}
	return result;
}

// remove-color.cpp ：去色滤镜
cv::Mat rColorFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			int max = std::max(
				std::max(img.at<cv::Vec3b>(i, j)[0], img.at<cv::Vec3b>(i, j)[1]),
				img.at<cv::Vec3b>(i, j)[2]
			);

			int min = std::min(
				std::min(img.at<cv::Vec3b>(i, j)[0], img.at<cv::Vec3b>(i, j)[1]),
				img.at<cv::Vec3b>(i, j)[2]
			);

			for (size_t k = 0; k < 3; k++)
			{
				result.at<cv::Vec3b>(i, j)[k] = (max + min) / 2;
			}
		}
	}
	return result;
}

// single-color-filter.cpp ：单色滤镜
cv::Mat sColorFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			result.at<cv::Vec3b>(i, j)[2] = 0;// red
			result.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];// green
			result.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];// blue
		}
	}
	return result;
}

// vintage-filter.cpp: 怀旧滤镜
cv::Mat vintageFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(0.349 * img.at<cv::Vec3b>(i, j)[2] + 0.686 * img.at<cv::Vec3b>(i, j)[1] + 0.168 * img.at<cv::Vec3b>(i, j)[0]);// green
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(0.393 * img.at<cv::Vec3b>(i, j)[2] + 0.769 * img.at<cv::Vec3b>(i, j)[1] + 0.189 * img.at<cv::Vec3b>(i, j)[0]);// red
			result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(0.272 * img.at<cv::Vec3b>(i, j)[2] + 0.534 * img.at<cv::Vec3b>(i, j)[1] + 0.131 * img.at<cv::Vec3b>(i, j)[0]);// blue
		}
	}
	return result;
}


// casting-filter.cpp: 熔铸滤镜
cv::Mat castingFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(128 * img.at<cv::Vec3b>(i, j)[0] / (img.at<cv::Vec3b>(i, j)[1] + img.at<cv::Vec3b>(i, j)[2] + 1));// blue
			result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(128 * img.at<cv::Vec3b>(i, j)[1] / (img.at<cv::Vec3b>(i, j)[0] + img.at<cv::Vec3b>(i, j)[2] + 1));// green
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(128 * img.at<cv::Vec3b>(i, j)[2] / (img.at<cv::Vec3b>(i, j)[0] + img.at<cv::Vec3b>(i, j)[1] + 1));// red
		}
	}
	return result;
}


// frozen-filter.cpp : 冰冻滤镜
cv::Mat frozenFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(std::abs(img.at<cv::Vec3b>(i, j)[0] - img.at<cv::Vec3b>(i, j)[1] - img.at<cv::Vec3b>(i, j)[2]) * 3 >> 1);// blue
			result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(std::abs(img.at<cv::Vec3b>(i, j)[1] - img.at<cv::Vec3b>(i, j)[0] - img.at<cv::Vec3b>(i, j)[2]) * 3 >> 1);// green
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(std::abs(img.at<cv::Vec3b>(i, j)[2] - img.at<cv::Vec3b>(i, j)[0] - img.at<cv::Vec3b>(i, j)[1]) * 3 >> 1);// red
		}
	}
	return result;
}


// comic-filter.cpp:连环画滤镜
cv::Mat comicFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			int r = img.at<cv::Vec3b>(i, j)[2];
			int g = img.at<cv::Vec3b>(i, j)[1];
			int b = img.at<cv::Vec3b>(i, j)[0];

			double R = std::abs(g - b + g + r) * r / 256;
			double G = std::abs(b - g + b + r) * r / 256;
			double B = std::abs(b - g + b + r) * g / 256;

			result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(B);//防止溢出（超过255）
			result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(G);
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(R);
		}
	}

	return result;
}

// relief-filter.cpp : 刻雕滤镜
cv::Mat relief2Filter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);

	for (size_t i = 1; i < img.rows - 1; i++) {
		for (size_t j = 1; j < img.cols - 1; j++) {
			for (size_t k = 0; k < 3; k++) {
				result.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(img.at<cv::Vec3b>(i, j)[k] - img.at<cv::Vec3b>(i - 1, j - 1)[k] + 128);
			}
		}
	}
	return result;
}

// relief - filter.cpp : 浮雕滤镜
cv::Mat relief1Filter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);

	for (size_t i = 1; i < img.rows - 1; i++) {
		for (size_t j = 1; j < img.cols - 1; j++) {
			for (size_t k = 0; k < 3; k++) {
				result.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(img.at<cv::Vec3b>(i + 1, j + 1)[k] - img.at<cv::Vec3b>(i - 1, j - 1)[k] + 128);
			}
		}
	}
	return result;
}


// drawing-filter.cpp : 素描滤镜
cv::Mat drawingFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);

	cv::Mat gray(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++) {
		for (size_t j = 0; j < img.cols; j++) {
			int max = std::max(std::max(img.at<cv::Vec3b>(i, j)[0], img.at<cv::Vec3b>(i, j)[1]), img.at<cv::Vec3b>(i, j)[2]);
			int min = std::min(std::min(img.at<cv::Vec3b>(i, j)[0], img.at<cv::Vec3b>(i, j)[1]), img.at<cv::Vec3b>(i, j)[2]);
			for (size_t k = 0; k < 3; k++) {
				gray.at<cv::Vec3b>(i, j)[k] = (max + min) / 2;
			}
		}
	}

	cv::Mat gray_revesal(img.size(), CV_8UC3);
	for (size_t i = 0; i < img.rows; i++) {
		for (size_t j = 0; j < img.cols; j++) {
			for (size_t k = 0; k < 3; k++) {
				gray_revesal.at<cv::Vec3b>(i, j)[k] = 255 - gray.at<cv::Vec3b>(i, j)[k];
			}
		}
	}

	cv::GaussianBlur(gray_revesal, gray_revesal, cv::Size(5, 5), 3);

	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				int a = gray.at<cv::Vec3b>(i, j)[k];
				int b = gray_revesal.at<cv::Vec3b>(i, j)[k];
				int c = std::min(a + (a * b) / (255 - b), 255);

				result.at<cv::Vec3b>(i, j)[k] = c;
			}
		}
	}

	return result;
}


// feather - filter.cpp : 羽化
cv::Mat featherFilter(cv::Mat img) {
	cv::Mat result(img.size(), CV_8UC3);
	int center_x = img.cols >> 1;
	int center_y = img.rows >> 1;
	int s2 = center_x * center_x + center_y * center_y;

	//宽长比例ratio
	double ratio = img.cols > img.rows ? static_cast<double>(img.rows) / img.cols : static_cast<double>(img.cols) / (img.rows);
	//控制V值得大小实现范围控制
	double mSize = 0.5;

	for (size_t i = 0; i < img.rows; i++) {
		for (size_t j = 0; j < img.cols; j++) {
			double dx = static_cast<double>(std::abs(center_x - static_cast<int>(j)));
			double dy = static_cast<double>(std::abs(center_y - static_cast<int>(i)));

			if (center_x > center_y) {
				dx = dx * ratio;
			}
			else {
				dy = dx * ratio;
			}

			double s1 = dx * dx + dy * dy;
			// V = 255 * 当前点Point距中点距离的平方s1 / (顶点距中点的距离平方s2 * mSize);
			double v = 255 * s1 / (s2 * mSize);

			int b = img.at<cv::Vec3b>(i, j)[0];
			int g = img.at<cv::Vec3b>(i, j)[1];
			int r = img.at<cv::Vec3b>(i, j)[2];

			result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(b + v);
			result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(g + v);
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(r + v);

		}
	}

	return result;
}

//ps-diffusion-filter.cpp: ps扩散特效
cv::Mat psDiffusionFilter(cv::Mat img) {
	//random engine | 随机数引擎
	std::default_random_engine generator;
	std::uniform_int_distribution<int> dis(1, 8);

	cv::Mat result(img.size(), CV_8UC3);

	for (size_t i = 1; i < img.rows - 1; i++) {
		for (size_t j = 1; j < img.cols - 1; j++) {
			int r = dis(generator);

			switch (r) {
			case 1:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i - 1, j - 1)[k];
				}
				break;
			case 2:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i - 1, j)[k];
				}
				break;
			case 3:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i - 1, j + 1)[k];
				}
				break;
			case 4:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i, j - 1)[k];
				}
				break;
			case 5:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i, j + 1)[k];
				}
				break;
			case 6:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i + 1, j - 1)[k];
				}
				break;
			case 7:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i + 1, j)[k];
				}
				break;
			case 8:
				for (size_t k = 0; k < 3; k++) {
					result.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(i + 1, j + 1)[k];
				}
				break;
			default:
				assert(false);
				break;
			}
		}
	}
	return result;
}

// opencv创建晕影滤镜
// 帮助器函数计算 2 点之间的距离。
double dist(cv::Point a, cv::Point b)
{
	return sqrt(pow((double)(a.x - b.x), 2) + pow((double)(a.y - b.y), 2));
}

// 帮助器函数,用于计算从边缘到中心点最远的距离。
double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center)
{
	// given a rect and a line | 给定一个矩形和一条线
	// get which corner of rect is farthest from the line | 得到哪个角的矩形是离线最远


	std::vector<cv::Point> corners(4);
	corners[0] = cv::Point(0, 0);
	corners[1] = cv::Point(imgSize.width, 0);
	corners[2] = cv::Point(0, imgSize.height);
	corners[3] = cv::Point(imgSize.width, imgSize.height);

	double max_dis = 0;
	for (int i = 0; i < 4; ++i)
	{
		double dis = dist(corners[i], center);
		if (max_dis < dis)
			max_dis = dis;
	}

	return max_dis;
}

// 帮助函数用于创建一个渐变的图像
// first_point,半径和功率是控制滤波器艺术效果的变量。

void generateGradient(cv::Mat& mask)
{
	cv::Point first_point = cv::Point(mask.size().width / 2, mask.size().height / 2);
	double radius = 1.0;
	double power = 0.6;

	// max image radian | 最大图像半径
	double max_image_rad = radius * getMaxDisFromCorners(mask.size(), first_point);

	mask.setTo(cv::Scalar(1));
	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{
			double temp = dist(first_point, cv::Point(j, i)) / max_image_rad;
			temp = temp * power;
			double temp_s = pow(cos(temp), 4);
			mask.at<double>(i, j) = temp_s;
		}
	}
}



int main()
{
	cv::Mat image = cv::imread("");
	cv::Mat result;
	if (!image.data)
	{
		std::cout << "failed to read the image!" << std::endl;
		return -1;
	}

	result = bwFilter(image);
	cv::imshow("黑白滤镜", result);

	result = drawingFilter(image);
	cv::imshow("素描滤镜", result);

	cv::waitKey(0);

	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
