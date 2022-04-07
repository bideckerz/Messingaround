#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono> 
#include "opencv2/cudastereo.hpp"

using namespace cv;
using namespace std;

void gaussianConv(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_gaussConv, int kernelSize, int sigma);



int ùndistort(int argc, char** argv)
{
    Mat img1, img2, dst1, dst2, lft, rgt;
    lft = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\left.bmp", IMREAD_COLOR);
    rgt = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\right.bmp", IMREAD_COLOR);


    Mat h_lft, h_rgt;
    cuda::GpuMat d_left, d_right;
    Ptr<cuda::StereoBM> bm;
    Ptr<cuda::StereoBeliefPropagation> bp;
    Ptr<cuda::StereoConstantSpaceBP> csbp;

    


    /////////////////////CUDA GAUSSIAN//////////////////////////////////////////////////////////////////////////////
    

    Mat h_img = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\right.bmp", IMREAD_COLOR); 

    
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;
    cv::cuda::GpuMat d_kernel;
    cv::Mat_<float> h_kernel;
    
    cv::Mat_<cv::Vec3b> input_img;
    cv::Mat_<cv::Vec3b> h_result;
    
    const int kernelSize = 9;
    int sigma = 3;
    
    int border = (int)(kernelSize - 1) / 2;
    
    cv::copyMakeBorder(h_img, input_img, border, border, border, border, cv::BORDER_REPLICATE);
    
    //h_kernel = gaussMat(kernelSize, sigma);
    d_kernel.upload(h_kernel);
    
    d_img.upload(input_img);
    d_result.upload(input_img);


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<double>* lintrin_arr = new std::vector<double>{ 6617.832736895683, 0, 551.3149037466272, 0, 6653.690602126069, 556.9070966245184, 0, 0, 1 };
    std::vector<double>* rintrin_arr = new std::vector<double>{ 6865.490775742195, 0, 679.9317259044155, 0, 6931.301017729188, 585.2658556698447, 0, 0, 1 };
    cv::Mat lk = cv::Mat(3, 3, CV_64F, lintrin_arr->data());
    cv::Mat rk = cv::Mat(3, 3, CV_64F, rintrin_arr->data());

    Mat rdist(1, 4, cv::DataType<double>::type);
    rdist.at<double>(0, 0) = 0;
    rdist.at<double>(0, 1) = 0;
    rdist.at<double>(0, 2) = -0.08920547497423885;
    rdist.at<double>(0, 3) = -0.08353444664858659;

    Mat lXi(1, 1, cv::DataType<double>::type);
    lXi.at<double>(0, 0) = 7.125969950873118;


    Mat ldist(1, 4, cv::DataType<double>::type);
    ldist.at<double>(0, 0) = 0;
    ldist.at<double>(0, 1) = 0;
    ldist.at<double>(0, 2) = -0.04762313500199899;
    ldist.at<double>(0, 3) = -0.04514176028124323;

    Mat rXi(1, 1, cv::DataType<double>::type);
    rXi.at<double>(0, 0) = 7.496757778551663;

    Mat brvecs(3, 1, cv::DataType<double>::type);
    Mat btvecs(3, 1, cv::DataType<double>::type);
    brvecs.at<double>(0, 0) = 0.02258493181867122;
    brvecs.at<double>(1, 0) = -0.001871371100708675;
    brvecs.at<double>(2, 0) = 0.001595470672660145;

    btvecs.at<double>(0, 0) = 63.03479390789293;
    btvecs.at<double>(1, 0) = 0.08430347221701619;
    btvecs.at<double>(2, 0) = 3.720453315014276;


    print(lk);
    print(rk);
    print(rdist);
    print(ldist);
    print(rXi);
    print(lXi);
    print(brvecs);
    print(btvecs);
   


    //Mat imgSize;
    cv::Size imgSize = lft.size();


    Mat E = Mat::eye(3, 3, cv::DataType<double>::type);

    int flag = (cv::omnidir::CALIB_FIX_SKEW + cv::omnidir::CALIB_FIX_K1 + cv::omnidir::CALIB_FIX_K2);
    cv::Matx33d KNew(imgSize.width / 4, 0, imgSize.width / 2,
                     0, imgSize.height / 4, imgSize.height / 2,
                     0, 0, 1);
    Mat pointCloud;
    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-8);
    //double rms;
    Mat undistortedl, undistortedr;
    cout << "UndistortingL" << endl;
    cv::Matx33d Knew(imgSize.width / 3.1415, 0, 0,
        0, imgSize.height / 3.1415, 0,
        0, 0, 1);

    //omnidir::undistortImage(lft, undistortedl, lk, ldist, lXi, cv::omnidir::RECTIFY_PERSPECTIVE, KNew, imgSize, E);
    //omnidir::undistortImage(rgt, undistortedr, rk, rdist, rXi, cv::omnidir::RECTIFY_PERSPECTIVE, KNew, imgSize, E);


    //omnidir::undistortImage(lft, undistortedl, lk, ldist, lXi, cv::omnidir::RECTIFY_CYLINDRICAL, KNew, imgSize, E);
    //omnidir::undistortImage(rgt, undistortedr, rk, rdist, rXi, cv::omnidir::RECTIFY_CYLINDRICAL, Knew, imgSize, E);


    omnidir::undistortImage(lft, undistortedl, lk, ldist, lXi, cv::omnidir::RECTIFY_LONGLATI, Knew, imgSize, E);
    omnidir::undistortImage(rgt, undistortedr, rk, rdist, rXi, cv::omnidir::RECTIFY_LONGLATI, Knew, imgSize, E);
 
    ////////////////////////////////////////////////////////////////////////////////////////

    cvtColor(undistortedl, h_lft, COLOR_BGR2GRAY);
    cvtColor(undistortedr, h_rgt, COLOR_BGR2GRAY);

    d_left.upload(h_lft);
    d_right.upload(h_rgt);

    bm = cuda::createStereoBM(16 * 5, 19);
    bp = cuda::createStereoBeliefPropagation(64, 5, 5, CV_32FC1);
    csbp = cuda::createStereoConstantSpaceBP(128, 8, 4, 4, CV_32FC1);


    bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);

    Mat bmdisp(h_lft.size(), CV_8U);
    cuda::GpuMat d_bmdisp(h_lft.size(), CV_8U);
    bm->compute(d_left, d_right, d_bmdisp);
    d_bmdisp.download(bmdisp);
    imshow("Original ImageL", undistortedl);
    imshow("Original ImageR", undistortedr);
    imshow("disparity", bmdisp);
    imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\bmdispss.png", bmdisp);




    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int numDisparities = 16 * 8;
    int SADWindowSize = 5;
    cv::Mat disMap;
    int pointType = omnidir::XYZRGB;
    Mat imageRec1, imageRec2;
    
    cv::omnidir::stereoReconstruct(lft, rgt, lk, ldist, lXi, rk, rdist, rXi,brvecs, btvecs, cv::omnidir::RECTIFY_LONGLATI, numDisparities, SADWindowSize, disMap, imageRec1, imageRec2, cv::Size(1152, 1152), Knew, pointCloud, pointType);


    cv::omnidir::stereoReconstruct(lft, rgt, lk, ldist, lXi, rk, rdist, rXi, brvecs, btvecs, cv::omnidir::RECTIFY_PERSPECTIVE, numDisparities, SADWindowSize, disMap, imageRec1, imageRec2, cv::Size(1152, 1152), KNew, pointCloud, pointType);
    imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\disMapPers.png", disMap);


    cv::Mat map1, map2, map3, map4, undistort;

    int blockSize = 5;
    int preFilterType = 1;
    int preFilterSize = 1;
    int preFilterCap = 31;
    int minDisparity = 0;
    int textureThreshold = 10;
    int uniquenessRatio = 15;
    int speckleRange = 0;
    int speckleWindowSize = 0;
    int disp12MaxDiff = -1;
    int dispType = CV_16S;

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

    Mat imgL_gray, imgr_gray, disp,dispConvert;
    cv::cvtColor(imageRec1, imgL_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRec2, imgr_gray, cv::COLOR_BGR2GRAY);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;
    for (int i = 0; i < iter; i++)
    {
        //gaussianConv(d_img, d_result, 32, 32, d_kernel, kernelSize, sigma);
    }
    d_result.download(h_result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;
    cout << diff.count() << endl;
    cout << diff.count() / iter << endl;
    cout << iter / diff.count() << endl;
    
    d_result = d_result(cv::Range(border + 1, d_result.rows+1 - border), cv::Range(border + 1, d_result.cols+1 - border)).clone();
    
    //cv::imshow("Processed Image", h_result);

    //////////////////////////////////////////////////////////////


    
    //namedWindow("lft 1", WINDOW_AUTOSIZE); // Create a window for display.
    //imshow("lft 1", undistortedl); // Show our image inside it.
    //
    //namedWindow("Result 1", WINDOW_AUTOSIZE); // Create a window for display.
    //imshow("Result 1", imageRec1);
    //
    //namedWindow("Result 2", WINDOW_AUTOSIZE); // Create a window for display.
    //imshow("Result 2", imageRec2);
    //namedWindow("disparity", WINDOW_AUTOSIZE);
    //imshow("disparity", disMap);
    //namedWindow("POINTCLOUD", WINDOW_AUTOSIZE);
    //imshow("POINTCLOUD", pointCloud);
    waitKey(0); // Wait for a keystroke in the window
    return 0;



}
