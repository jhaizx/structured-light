// Example 18-1. Reading a chessboard’s width and height, reading and collecting
// the requested number of views, and calibrating the camera
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

void help(char** argv) {  // todo rewrite this
	cout << "\n\n"
		<< "Example 18-1:\nReading a chessboard's width and height,\n"
		<< "              reading and collecting the requested number of views,\n"
		<< "              and calibrating the camera\n\n"
		<< "Call:\n" << argv[0] << " <board_width> <board_height> <number_of_boards> <if_video,_delay_between_framee_capture> <image_scaling_factor>\n\n"
		<< "Example:\n" << argv[0] << " 9 6 15 500 0.5\n"
		<< "-- to use the checkerboard9x6.png provided\n\n"
		<< " * First it reads in checker boards and calibrates itself\n"
		<< " * Then it saves and reloads the calibration matricies\n"
		<< " * Then it creates an undistortion map and finally\n"
		<< " * It displays an undistorted image\n"
		<< endl;
}

void saveDisp(const char* filename, const cv::Mat& mat)
{
	errno_t err;
	FILE* fp;
	err = fopen_s(&fp, filename, "wt");
	fprintf(fp, "%02d/n", mat.rows);
	fprintf(fp, "%02d/n", mat.cols);
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			short disp = mat.at<short>(y, x); // 这里视差矩阵是CV_16S 格式的，故用 short 类型读取
			fprintf(fp, "%d/n", disp); // 若视差矩阵是 CV_32F 格式，则用 float 类型读取
		}
	}
	fclose(fp);
}

int main(int argc, char* argv[]) {
	float image_sf = 0.5f;

	cv::Mat image1, image2, image1r, image2r;
	image1 = cv::imread("IR_left.bmp", 0);
	image2 = cv::imread("IR_right.bmp", 0);

	cout << "读取相机内外参..." << endl;
	cv::Mat M1, M2, D1, D2, R, T;
	//必须用double而不是float
	M1 = (cv::Mat_<double>(3, 3) << 951.992, 0, 635.942
		, 0, 951.992, 398.72
		, 0, 0, 1);
	M2 = (cv::Mat_<double>(3, 3) << 956.774, 0, 638.843
		, 0, 956.774, 392.215
		, 0, 0, 1);
	D1 = (cv::Mat_<double>(1, 5) << 0.03655534, -0.1476006, 0.001574314, 0.0005308038, 0.0750581);
	D2 = (cv::Mat_<double>(1, 5) << 0.02833917, -0.1292386, 0.0006758662, 9.279677e-05, 0.05397161);
	R = (cv::Mat_<double>(3, 3) << 0.999999, 0.0006877045, 0.001258215
		, -0.0006886553, 0.9999995, 0.0007554362
		, -0.001257694, -0.0007563019, 0.9999989);
	T = (cv::Mat_<double>(3, 1) << -39.97538, -0.1411242, 0.1665998);
	cv::Size imageSize = image1.size();

	cout << "校正畸变" << endl;
	//使用Bouguet算法进行立体校正
	cv::Mat R1, R2, P1, P2, Q;
	stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2,
		Q, 0);
	cv::Mat map11, map12, map21, map22;
	cv::initUndistortRectifyMap(M1, D1,
		R1, P1, imageSize,
		CV_16SC2, map11, map12);
	cv::initUndistortRectifyMap(M2, D2,
		R2, P2, imageSize,
		CV_16SC2, map21, map22);



	if (image1.empty() && image2.empty()) {
		cout << "打开图片失败" << endl;
		return 0;
	}
	cv::remap(image1, image1r, map11, map12, cv::INTER_LINEAR,
		cv::BORDER_CONSTANT, cv::Scalar());
	cv::remap(image2, image2r, map21, map22, cv::INTER_LINEAR,
		cv::BORDER_CONSTANT, cv::Scalar());
	resize(image1r, image1r, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);
	resize(image2r, image2r, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);


	cv::Mat pair;
	pair.create(image1r.rows, image1r.cols * 2, CV_8UC3);


	cv::Mat part;
	part = pair.colRange(0, imageSize.width / 2);
	cv::cvtColor(image1r, part, cv::COLOR_GRAY2BGR);
	part = pair.colRange(imageSize.width / 2, imageSize.width);
	cv::cvtColor(image2r, part, cv::COLOR_GRAY2BGR);

	for (int j = 0; j < imageSize.height; j += (imageSize.height / 35))
		cv::line(pair, cv::Point(0, j), cv::Point(imageSize.width, j),
			cv::Scalar(0, 255, 0));
	
	imshow("对齐校正后的结果显示", pair);

	cout << "开始使用SGBM匹配算法进行匹配..." << endl;
	cv::Mat disp1, disp2, vdisp;
  //（最小视差）（）
	cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
		-64, 128, 11, 100, 1000, 32, 0, 15, 1000, 16, cv::StereoSGBM::MODE_HH);
	clock_t start, end;
	start = clock();
	stereo->compute(image1r, image2r, disp1);
	stereo->compute(image2r, image1r, disp2);
	cv::normalize(disp1, disp1, 0, 256, cv::NORM_MINMAX, CV_8U);
	cv::normalize(disp2, disp2, 0, 256, cv::NORM_MINMAX, CV_8U);

	pair.create(image1r.rows, image1r.cols * 2, CV_8UC3);
	part = pair.colRange(0, imageSize.width / 2);
	cv::cvtColor(disp1, part, cv::COLOR_GRAY2BGR);
	part = pair.colRange(imageSize.width / 2, imageSize.width);
	cv::cvtColor(disp2, part, cv::COLOR_GRAY2BGR);
	cv::imshow("disparity", pair);

	end = clock();
	cout << CLOCKS_PER_SEC << endl;
	cout << "匹配耗时：" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

	cv::normalize(disp2, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);	//取最大最小视差做归一化处理
	cv::imshow("disparity", vdisp);
	
	cv::Mat depth;
	cv::reprojectImageTo3D(vdisp, depth, Q, false, -1);
	cv::imshow("depth", depth);

	cv::waitKey(0);
	return 0;
}
