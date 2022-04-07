#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdlib.h> 

#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/calib3d.hpp> 


using namespace cv;
using namespace std;

int calib()
{
	vector< vector< Vec3f > > object_points;
	vector< vector< Vec2f > > limage_points, rimage_points;
	vector< Vec2f > cornersl, cornersr;
	int board_width = 9;
	int board_height = 6;
	int square_size = 20;
	cv::String pathL("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\omnidirimage\\Left\\*.bmp"); //select only jpg
	cv::String pathR("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\omnidirimage\\Right\\*.bmp"); //select only jpg
	Mat dist1 = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\omnidirimage\\Left\\left_stid1_frm95160_bid1.bmp", IMREAD_COLOR);
	Mat dist2 = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\omnidirimage\\Right\\right_stid1_frm95160_bid2.bmp", IMREAD_COLOR);

	vector<cv::String> fn,fnl;
	vector<cv::Mat> data;
	vector<int> lindex, rindex;
	cv::glob(pathR, fn , true);
	cv::glob(pathL, fnl, false);
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat imr = imread(fn[k], IMREAD_GRAYSCALE);
		Mat iml = imread(fnl[k], IMREAD_GRAYSCALE);
		if (iml.empty()) continue; //only proceed if successful
		// you probably want to do some preprocessing

		Size board_size = Size(board_width, board_height);
		int board_n = board_width * board_height;
		bool foundl = false;
		bool foundr = false;

		foundl = cv::findChessboardCorners(iml, board_size, cornersl,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
		foundr = cv::findChessboardCorners(imr, board_size, cornersr,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
		if (foundl && foundr)
		{
			cornerSubPix(iml, cornersl, cv::Size(11, 11), cv::Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 2000, 0.001));
			drawChessboardCorners(iml, board_size, cornersl, foundl);
			cornerSubPix(imr, cornersr, cv::Size(11, 11), cv::Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 2000, 0.001));
			drawChessboardCorners(imr, board_size, cornersr, foundr);
		}
		vector< Vec3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Vec3f((float)j * square_size, (float)i * square_size, 0));

		if (foundl && foundr) {
			cout << k << ". Found corners!" << endl;
			rimage_points.push_back(cornersl);
			object_points.push_back(obj);
			limage_points.push_back(cornersr);
			//robject_points.push_back(obj);
			//rindex.push_back(k);
			//data.push_back(iml);

		}

		//imshow("image", im);
		//waitKey(200);

	}
	 // recurse
	/*
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat im = imread(fn[k], IMREAD_GRAYSCALE);
		if (im.empty()) continue; //only proceed if successful
		// you probably want to do some preprocessing

		Size board_size = Size(board_width, board_height);
		int board_n = board_width * board_height;
		bool found = false;

		found = cv::findChessboardCorners(im, board_size, corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
		if (found)
		{
			cornerSubPix(im, corners, cv::Size(11, 11), cv::Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 2000, 0.001));
			drawChessboardCorners(im, board_size, corners, found);
		}
		vector< Vec3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Vec3f((float)j * square_size, (float)i * square_size, 0));

		if (found) {
			cout << k << ". Found corners!" << endl;
			for (auto h = rindex.begin(); h != rindex.end(); h++) {
				if (k == *h) {
					cout << *h<<endl;
					rimage_points.push_back(corners);
					robject_points.push_back(obj);
					data.push_back(im);

				}
			}


		}

		//imshow("image", im);
		//waitKey(200);
		data.push_back(im);

	}*/

	destroyAllWindows;

	cv::Mat lK, lxi, lD, lidx;
	cv::Mat rK, rxi, rD, ridx;
	cv::Mat idx;

	int flags = cv::omnidir::CALIB_FIX_SKEW + cv::omnidir::CALIB_FIX_K1 + cv::omnidir::CALIB_FIX_K2;
	//int flags = 0;
	int flagf = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW;
	cv::TermCriteria critia (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-8);
	//cv::TermCriteria critia2(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON);

	Mat lrvecs, ltvecs;
	Mat rrvecs, rtvecs;
	Mat brvecs, btvecs;





	

	cv::Size imgSize = dist1.size();
	int numDisparities = 16 * 5;
	int SADWindowSize = 5;
	cv::Mat disMap;
	int flag = cv::omnidir::RECTIFY_PERSPECTIVE;
	int pointType = omnidir::XYZRGB;
	// the range of theta is (0, pi) and the range of phi is (0, pi)
	cv::Matx33d KNew(imgSize.width / 4, 0, imgSize.width / 2, 0, imgSize.height / 4, imgSize.height / 2, 0, 0, 1);
	Mat imageRec1, imageRec2, pointCloud;
	//double rms = cv::fisheye::calibrate(object_points, image_points, cv::Size(1280, 960), k, d, rvecs, tvecs, flagf, critia2);
	cout<<"Calibrating"<<endl;
	double rmsl = cv::omnidir::calibrate(object_points, limage_points, cv::Size(1152, 1152), lK, lxi, lD, lrvecs, ltvecs, flag, critia, lidx);
	cout<<"Done with left" << endl;
	double rmsr = cv::omnidir::calibrate(object_points, rimage_points, cv::Size(1152, 1152), rK, rxi, rD, rrvecs, rtvecs, flag, critia, ridx);
	cout << "Done with right" << endl;
	cout << "Stereo Calibrating" << endl;
	double rms = cv::omnidir::stereoCalibrate(object_points, limage_points, rimage_points, imgSize, imgSize, lK, lxi, lD, rK, rxi, rD, brvecs, btvecs, lrvecs, ltvecs, flag, critia, idx);
	print(lK);
	print(rK);
	print(rD);
	print(lD);
	print(rxi);
	print(lxi);
	print(brvecs);
	print(btvecs);
	print(lrvecs);
	print(rrvecs);
	print(rtvecs);
	print(ltvecs);

	Mat undistortedl, undistortedr;
	//imshow("undistorted", dist);
	//waitKey(1000);

	cout << "UndistortingL" << endl;

	omnidir::undistortImage(dist1, undistortedl, lK, lD, lxi, flag, KNew, cv::Size(1152, 1152));
	cout << "UndistortingR" << endl;

	omnidir::undistortImage(dist2, undistortedr, rK, rD, rxi, flag, KNew, cv::Size(1152, 1152));

	cv::omnidir::stereoReconstruct(undistortedl, undistortedr, lK, lD, lxi, rK, rD, rxi, brvecs, btvecs, flag, numDisparities, SADWindowSize, disMap, imageRec1, imageRec2, imgSize, KNew, pointCloud, pointType);

	//fisheye::undistortImage(dist, undistort, k,d,k, cv::Size(1500, 800));

	imshow("DisMap", disMap);
	imshow("undistorted", imageRec1);
	imshow("undistorted", imageRec2);
	waitKey(0);

	return 0;
}