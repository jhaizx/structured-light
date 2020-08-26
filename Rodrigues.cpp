#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc,char** argv) {
	float v[3] = { 0,1,0 };
	float R[3][3];
	cv::Mat mat_R;
	cv::Mat mat_V(1, 3, CV_32FC1, v);
	cv::Mat pian;
	cv::Rodrigues(mat_V, mat_R, pian);
	std::cout << mat_V << std::endl;
	std::cout << mat_R << std::endl;
	std::cout << pian << std::endl;
}
