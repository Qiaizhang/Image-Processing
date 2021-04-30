#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/calib3d.hpp>


using namespace std;
using namespace cv;

const double EPS = 0.4;

Mat getHoGKernel(Size ksize, double sigma)
{
	Mat kernel(ksize, CV_64F);
	Point centPoint = Point((ksize.width - 1) / 2, (ksize.height - 1) / 2);
	// first calculate Gaussian
	for (int i = 0; i < kernel.rows; i++) {
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++) {
			double param = -((i - centPoint.y) * (i - centPoint.x)) / (2 * sigma * sigma);
			pData[j] = exp(param);
		}
	}
	double maxValue;
	minMaxLoc(kernel, NULL, &maxValue);//Ñ°ÕÒ×î´óÖµ
	for (int i = 0; i < kernel.rows; i++) {
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++) {
			if (pData[j] < EPS*maxValue) {
				pData[j] = 0;
			}
		}
	}
	
	double sumKernel = sum(kernel)[0];
	
	if (sumKernel != 0) {
		kernel = kernel / sumKernel;
	}
	// now calculate Laplacian
	for (int i = 0; i < kernel.rows; i++) {	
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++) {
			double addition = ((i - centPoint.y) * (i - centPoint.y) + (j - centPoint.x) * (j - centPoint.x) - 2 * sigma*sigma) / (sigma * sigma * sigma * sigma);
			pData[j] *= addition;                                                 
		}
	}

	sumKernel = sum(kernel)[0];
	kernel -= (sumKernel / ksize.width * ksize.height);

	return kernel;
}


//int main() {
//	
//	//Mat kernel = getHoGKernel(Size(3,3), 0.5);
//
//	Mat image = imread("D:\\Image\\13.png");
//	vector<KeyPoint> keyPoints;
//	SimpleBlobDetector::Params params;
//	Ptr<SimpleBlobDetector>blobDetect = SimpleBlobDetector::create(params); 
//	blobDetect->detect(image, keyPoints);
//
//	cout << keyPoints.size() << endl;
//	drawKeypoints(image, keyPoints, image, Scalar(255, 0, 0));
//
//	namedWindow("blobs");
//	imshow("blobs", image);
//	waitKey();
//	return 0;
//}