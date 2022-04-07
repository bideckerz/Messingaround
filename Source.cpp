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
#include <fstream>
#include <opencv2/cudaarithm.hpp>

#define KERNEL_RADIUS  32
#define RADIUS 32
#define KERNEL_LENGTH_X(x) (2 * x + 1)
#define MAX_KERNEL_LENGTH KERNEL_LENGTH(MAX_KERNEL_RADIUS)
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

using namespace cv;
using namespace std;

void copyKernel(float* kernel_coefficients, int kernel_index);
void testKernel();

void gaussianSep(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dm, cv::cuda::GpuMat& d_kernelGauss, int pass, float focus_depth);

void gaussianConv(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dm, int dimX, int dimY, cv::cuda::GpuMat& d_gaussConv, int kernelSize, int sigma);

void depthProcess(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, double fx, double bl, float scale, int doffs);

void normal(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int dimX, int dimY, int step, float min, float max);

void convolutionRows(uchar3* d_Dst, uchar3* d_Src, float* i_depth, int imageW, int imageH, size_t pitch, size_t depth_pitch, float focus_point);

void convolutionColumns(uchar3* d_Dst, uchar3* d_Src, float* i_depth, int imageW, int imageH, size_t pitch, size_t depth_pitch, float focus_point);

cv::Mat_<float> generateGaussianKernel1D(int kernelSize, int sigma)
{
    float PI = 3.14159265358979323846;
    float constant = 1.0 / (2.0 * PI * pow(sigma, 2));
    int rad = (kernelSize - 1.0) / 2.0;
    cv::Mat_<float> h_kernel(kernelSize, 1);


    float sum = 0.0;
    for (int i = -rad; i < rad + 1; i++) {
        h_kernel[i + rad][0] = constant * (exp(-pow(i, 2) / (2 * pow(sigma, 2))));
        sum += h_kernel[i + rad][0];
    }

    for (int i = 0; i < kernelSize; ++i) {
        h_kernel[i][0] /= sum;
    }

    return h_kernel;
}

class DepthOfFieldRenderer {
    vector<int> filter_sizes_;
    float kDepthStep;
public:
    DepthOfFieldRenderer(float _depth_step);
    ~DepthOfFieldRenderer();
    
    void RenderDoF(Point origin);
    
}; 





int main(int argc, char** argv)
{
    Mat  lft, rgt;

    Mat lftp(1920, 1080, CV_32FC1);
    Mat rgtp(1920, 1080, CV_32FC1);

    lft = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\im0.png", IMREAD_COLOR);
    rgt = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\im1.png", IMREAD_COLOR);

    lftp = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\disp0.pfm", IMREAD_UNCHANGED);
    rgtp = imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\disp1.pfm", IMREAD_UNCHANGED);


    


    cv::cuda::GpuMat d_left, d_right,d_pfml, d_pfmr, d_resultl, d_resultr;

    cv::cuda::GpuMat d_kernel;
    cv::Mat_<float> h_kernel;

    cv::cuda::GpuMat d_kernel1D;
    cv::Mat_<float> h_kernel1D;

    cv::Mat_<cv::Vec3b> input_imgl, input_imgr;

    //const int kernelSize = 11;
    //int sigma = 23;
    //
    //int border;
    
    const int kernelSize = 2 * RADIUS + 1;
    int sigma = 11;
    int border;
    
    (kernelSize % 2 == 0) ? border = kernelSize / 2 : border = (kernelSize - 1) / 2;

    cv::copyMakeBorder(lft, input_imgl, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(rgt, input_imgr, border, border, border, border, cv::BORDER_REPLICATE);
    /*
    vector<float> gauss_vec;
    // Create all the gaussien kernel for different radius and copy them to GPU
    for (int i = 0; i < kernelSize; i++) {
        gauss_vec.resize((i + 1) * 2 + 1, 0);

         //Compute Gaussian coeff
        int radd = (gauss_vec.size() - 1) / 2;
        float zigma = 0.3f * ((gauss_vec.size() - 1.f) * 0.5f - 1.f) + 0.8f;
        float zum = 0;
        for (int u = -radd; u <= radd; u++) {
            float gauss_value = expf(-1.f * (powf(u, 2.f) / (2.f * powf(zigma, 2.f))));
            gauss_vec[u + radd] = gauss_value;
            zum += gauss_value;
        }
        zum = 1.f / zum;
        for (int u = 0; u < gauss_vec.size(); u++) {
            gauss_vec[u] *= zum;
            cout << gauss_vec[u];
        }
            

        // Copy coeff to GPU
        copyKernel(gauss_vec.data(), i);
        cout << " "<< endl;
    }*/


    vector<int> filter_sizes_;
    for (int radius = 1; radius * 0.1 < 1.0; ++radius)
        filter_sizes_.push_back(2 * radius + 1);
    const char* empty;

    vector<Mat> gaussianKernel;
    vector<float*> h_kernel3;
    for (int i = 0; i < filter_sizes_.size(); ++i) {
        gaussianKernel.push_back(getGaussianKernel(filter_sizes_[i], -1, CV_32F));
        h_kernel3.push_back(gaussianKernel[i].ptr<float>(0));
        copyKernel(h_kernel3[i], i);
    }
    testKernel();

    

    
    h_kernel1D = generateGaussianKernel1D(kernelSize, sigma);
    //cout << h_kernel1D;
    d_kernel1D.upload(h_kernel1D);

    //h_kernel = gaussMat(kernelSize, sigma);
    //d_kernel.upload(h_kernel);

    d_left.upload(input_imgl);
    d_right.upload(input_imgr);

    d_resultl.upload(input_imgl);
    d_resultr.upload(input_imgl);
    
    double focalLenght = 1733;
    double baseline = 536.62;
    int width = 1920;
    int height = 1080;
    int ndisp = 170;
    int vmin = 55;
    int vmax = 142;
    int doffs = 0;
    float scale = 0.0039;

    Mat depthmapL(1920, 1080, CV_32FC1);
    Mat depthmapR(1920, 1080, CV_32FC1);
    Mat d_depthmapL(1920+border, 10800+border, CV_32FC1);
    Mat d_depthmapR(1920+border, 1080+border, CV_32FC1);

    //print(lft);
    //depthmap = (focalLenght * baseline / (lft / scale ));
    
    d_pfml.upload(lftp);
    d_pfmr.upload(rgtp);

    cv::cuda::GpuMat d_respfml;
    cv::cuda::GpuMat d_respfmr;

    d_respfml.upload(lftp);
    d_respfmr.upload(rgtp);

    depthProcess(d_pfml, d_respfml, 32, 32, focalLenght, baseline, scale, doffs);
    depthProcess(d_pfmr, d_respfmr, 32, 32, focalLenght, baseline, scale, doffs);

    cv::cuda::GpuMat d_depthl_norm;
    cv::cuda::GpuMat d_depthr_norm;


    d_respfml.download(depthmapL);
    d_respfmr.download(depthmapR);   

    uchar3* lftu, * rgtu, * h_outputL;
    uchar3* d_lftu, * d_rgtu, * d_outputL, *d_bufferImg;
    float* h_depthmapL, * d_depthL;

    while (!lft.isContinuous())
        lft = lft.clone();

    lftu = (uchar3*)lft.ptr<Vec3b>(0);
    rgtu = (uchar3*)rgt.ptr<Vec3b>(0);

    double minl, maxl;
    cv::minMaxLoc(depthmapL, &minl, &maxl);

    double minr, maxr;
    cv::minMaxLoc(depthmapR, &minr, &maxr);

    d_depthl_norm.upload(lftp);
    d_depthr_norm.upload(lftp);

    normal(d_respfml, d_depthl_norm, 32,32,1, minl,maxl);
    normal(d_respfmr, d_depthr_norm, 32, 32, 1, minr, maxr);


    Mat nor_depthmapL(1920, 1080, CV_32FC1);
    Mat nor_depthmapR(1920, 1080, CV_32FC1);

    d_depthl_norm.download(nor_depthmapL);
    d_depthr_norm.download(nor_depthmapR);
 
    
    cv::copyMakeBorder(nor_depthmapL, d_depthmapL, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(nor_depthmapR, d_depthmapR, border, border, border, border, cv::BORDER_REPLICATE);

    cv::cuda::GpuMat d_depthl;
    cv::cuda::GpuMat d_depthr;

    d_depthl.upload(d_depthmapL);
    d_depthr.upload(d_depthmapR);

    cv::cuda::GpuMat d_tmp_imgl;
    cv::cuda::GpuMat d_tmp_imgr;

    d_tmp_imgl.upload(input_imgl);
    d_tmp_imgr.upload(input_imgr);



    size_t pitch;
    cudaMallocPitch(&d_lftu, &pitch, width * sizeof(Vec3b), height);
    cudaMemcpy2D(d_lftu, pitch, lftu, width * sizeof(Vec3b), width * sizeof(Vec3b), height, cudaMemcpyHostToDevice);

    size_t depth_pitch;
    h_depthmapL = nor_depthmapL.ptr<float>(0);

    cudaMallocPitch(&d_depthL, &depth_pitch, width * sizeof(float), height);
    cudaMemcpy2D(d_depthL, depth_pitch, h_depthmapL, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);



    h_outputL = (uchar3*)malloc(width * height * sizeof(Vec3b));
    cudaMalloc((void**)&d_outputL, pitch * height);
    cout <<"pitch "<< pitch<<endl;
    cudaMemset(d_outputL, 0, pitch * height);

    cudaMalloc((void**)&d_bufferImg, pitch * height);
    float depth_of_focus = nor_depthmapL.at<float>(1000,1000);
    

    convolutionRows(lftu, d_bufferImg, d_depthL, width, height, pitch, depth_pitch, depth_of_focus);
    convolutionColumns(d_bufferImg, d_outputL, d_depthL, width, height,pitch, depth_pitch , depth_of_focus);
    
    cudaMemcpy2D(h_outputL, width * sizeof(Vec3b), d_outputL, pitch, width * sizeof(Vec3b), height, cudaMemcpyDeviceToHost);
    Mat output_image_color(lft.size(), lft.type(), h_outputL);
   

    //gaussianSep(d_left, d_tmp_imgl, d_depthl, d_kernel1D, 1,0.2);
    //gaussianSep(d_tmp_imgl, d_resultl, d_depthl, d_kernel1D, 0,0.2);

    //gaussianSep(d_right, d_tmp_imgr, d_depthr, d_kernel1D, 1, 0.2);
    //gaussianSep(d_tmp_imgr, d_resultr, d_depthr, d_kernel1D, 0, 0.2);

    //gaussianConv(d_left, d_resultl, d_depthl, 32, 32, d_kernel, kernelSize, sigma);
    //gaussianConv(d_right, d_resultr, d_depthr, 32, 32, d_kernel, kernelSize, sigma);

    cv::Mat_<cv::Vec3b> h_resultl;
    //d_resultl.download(h_resultl);
    //cv::Mat_<cv::Vec3b> h_resultr;
    //d_resultr.download(h_resultr);
    h_resultl = output_image_color;
    
    //cout << depthmapL;
    //imshow("disp1", lft);
    //imshow("disp2", rgt);
    //imshow("depthmapL",depthmapL);
    //imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\depthl.jpg", depthmapL);
    //imshow("depthmapR", depthmapR);
    //imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\depthr.jpg", depthmapR);
    
    imshow("resultl", h_resultl);
    //imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\depthblurl.png", h_resultl);
    //imshow("resultr", h_resultr);
    //imwrite("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\OpenCV\\results\\depthblurr.png", h_resultr);




    waitKey(-1);
    return 0;

}