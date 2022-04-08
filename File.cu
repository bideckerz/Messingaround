#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>


#include <assert.h>
#include <math.h>

#define MAX_KERNEL_RADIUS 9
#define NUM_KERNELS MAX_KERNEL_RADIUS
#define KERNEL_LENGTH(x) (2 * x + 1)
#define MAX_KERNEL_LENGTH KERNEL_LENGTH(MAX_KERNEL_RADIUS)

#define ROW_TILE_WIDTH 32
#define ROW_TILE_HEIGHT 4
#define ROW_TILES_IN_BLOCK 20
#define ROW_BLOCK_WIDTH ROW_TILES_IN_BLOCK * ROW_TILE_WIDTH
#define ROW_BLOCK_HEIGHT ROW_TILE_HEIGHT

__constant__ float c_kernel[NUM_KERNELS * (NUM_KERNELS + 2)];

void copyKernel(float* kernel_coefficients, int kernel_index) {
    int kernel_radius = kernel_index + 1;
    cudaMemcpyToSymbol(
        c_kernel,
        kernel_coefficients,
        KERNEL_LENGTH(kernel_radius) * sizeof(float),
        kernel_index * (kernel_index + 2) * sizeof(float));
}

void testKernel() {
    float h_kernel_data[NUM_KERNELS * (NUM_KERNELS + 2)];
    cudaMemcpyFromSymbol(h_kernel_data, c_kernel, NUM_KERNELS * (NUM_KERNELS + 2) * sizeof(float));
    int i, j;
    for (i = 0; i < NUM_KERNELS; ++i) {
        printf("%d: ", i);
        for (j = 0; j < 2 * i + 3; ++j)
            printf("%f ", h_kernel_data[i * (i + 2) + j]);
        printf("\n");
    }
}


__global__ void k_normalize(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int rows, int cols, int step, float min, float max)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows)
    {
        (src(dst_y, dst_x) == 0) ? dst(dst_y, dst_x) = 0 : dst(dst_y, dst_x) = (max - src(dst_y, dst_x)) / (max - min);
    }



}

__global__ void depth(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int rows, int cols, double fx, double bl, float scale, int doffs)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;


    if (dst_x < cols && dst_y < rows)
    {
        dst(dst_y, dst_x) = (fx * bl / (src(dst_y, dst_x) / scale + doffs));
    }
}


__global__ void convolveSeparableRowsKernel(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth) {
    __shared__ unsigned char s_data[ROW_TILE_HEIGHT][(ROW_BLOCK_WIDTH + 2 * ROW_TILE_WIDTH)];
    int x = threadIdx.x, y = threadIdx.y;
    int x_image, y_image, x_s, y_s;

    x_image = blockIdx.x * ROW_BLOCK_WIDTH - ROW_TILE_WIDTH + x;
    y_image = blockIdx.y * ROW_BLOCK_HEIGHT + y;
    x_s = x; y_s = y;
    s_data[y_s][x_s] = x_image < 0 ? 0 : d_src[y_image * pitch + x_image];

    for (int i = 1; i < (ROW_TILES_IN_BLOCK + 2); ++i) {
        x_s += ROW_TILE_WIDTH;
        x_image += ROW_TILE_WIDTH;
        s_data[y_s][x_s] = x_image >= image_width * 3 ? 0 : d_src[y_image * pitch + x_image];
    }
    __syncthreads();

    x_image = blockIdx.x * ROW_BLOCK_WIDTH + x;
    x_s = ROW_TILE_WIDTH + x;

    for (int i = 0; i < ROW_TILES_IN_BLOCK; ++i) {
        if (x_image < image_width * 3) {
            int kernel_radius = (int)floor(10 * fabs(d_depth_map[y_image * depth_map_pitch / sizeof(float) + x_image / 3] - focus_depth));
            if (kernel_radius > 0) {
                float sum = 0;
                int kernel_start = kernel_radius * kernel_radius - 1;
                int kernel_mid = kernel_start + kernel_radius;
                for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                    sum += (float)s_data[y_s][x_s + j * 3] * c_kernel[kernel_mid + j];
                }
                d_dst[y_image * pitch + x_image] = (unsigned char)sum;
            }
            else {
                d_dst[y_image * pitch + x_image] = s_data[y_s][x_s];
            }
        }
        x_s += ROW_TILE_WIDTH;
        x_image += ROW_TILE_WIDTH;
    }
}

 void GpuConvolveSeparableRows(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block_grid_width = (int)ceil((float)image_width * 3 / (ROW_TILES_IN_BLOCK * ROW_TILE_WIDTH));
    int block_grid_height = (int)ceil((float)image_height / (ROW_TILE_HEIGHT));
    printf("block_grid_width:%d block_grid_height:%d\n", block_grid_width, block_grid_height);
    printf("image_width:%d image_height:%d\n", image_width, image_height);
    dim3 blocks(block_grid_width, block_grid_height);
    dim3 threads(ROW_TILE_WIDTH, ROW_TILE_HEIGHT);
    cudaEventRecord(start, 0);

    convolveSeparableRowsKernel << <blocks, threads >> > (
        d_dst,
        d_src,
        d_depth_map,
        image_width,
        image_height,
        pitch,
        depth_map_pitch,
        focus_depth
        );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

}

#define COL_TILE_WIDTH 16
#define COL_TILE_HEIGHT 10
#define COL_VTILES_IN_BLOCK 8
#define COL_HTILES_IN_BLOCK 2
#define COL_BLOCK_WIDTH COL_TILE_WIDTH * COL_HTILES_IN_BLOCK
#define COL_BLOCK_HEIGHT COL_TILE_HEIGHT * COL_VTILES_IN_BLOCK 

__global__ void convolveSeparableColsKernel(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth) {
    __shared__ unsigned char s_data[COL_BLOCK_HEIGHT + 2 * COL_TILE_HEIGHT][COL_BLOCK_WIDTH];
    int x = threadIdx.x, y = threadIdx.y;
    int x_image, y_image, x_s, y_s;

    x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
    y_image = blockIdx.y * COL_BLOCK_HEIGHT - COL_TILE_HEIGHT + y;
    x_s = x; y_s = y;

    for (int k = 0; k < COL_HTILES_IN_BLOCK; ++k) {
        if (x_image < image_width * 3) {
            s_data[y_s][x_s] = y_image < 0 ? 0 : d_src[y_image * pitch + x_image];
            x_image += COL_TILE_WIDTH;
            x_s += COL_TILE_WIDTH;
        }
    }
    for (int i = 1; i < COL_VTILES_IN_BLOCK + 2; ++i) {
        x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
        x_s = x;
        y_s += COL_TILE_HEIGHT;
        y_image += COL_TILE_HEIGHT;
        for (int k = 0; k < COL_HTILES_IN_BLOCK; ++k) {
            if (x_image < image_width * 3) {
                s_data[y_s][x_s] = y_image < image_height ? d_src[y_image * pitch + x_image] : 0;
                x_image += COL_TILE_WIDTH;
                x_s += COL_TILE_WIDTH;
            }
        }
    }
    __syncthreads();
    x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
    x_s = x;
    for (int k = 0; k < COL_HTILES_IN_BLOCK; ++k) {
        if (x_image < image_width * 3) {
            y_image = blockIdx.y * COL_BLOCK_HEIGHT - COL_TILE_HEIGHT + y;
            y_s = y;

            for (int i = 0; i < COL_VTILES_IN_BLOCK; ++i) {
                y_s += COL_TILE_HEIGHT;
                y_image += COL_TILE_HEIGHT;
                if (y_image < image_height) {
                    int kernel_radius = (int)floor(10 * fabs(d_depth_map[y_image * depth_map_pitch / sizeof(float) + x_image / 3] - focus_depth));
                    if (kernel_radius > 0) {
                        float sum = 0;
                        int kernel_start = kernel_radius * kernel_radius - 1;
                        int kernel_mid = kernel_start + kernel_radius;
                        for (int j = -kernel_radius; j <= kernel_radius; ++j)
                            sum += (float)s_data[y_s + j][x_s] * c_kernel[kernel_mid + j];
                        d_dst[y_image * pitch + x_image] = (unsigned char)sum;
                    }
                    else
                        d_dst[y_image * pitch + x_image] = s_data[y_s][x_s];
                }
            }
        }
        x_image += COL_TILE_WIDTH;
        x_s += COL_TILE_WIDTH;
    }
}

 void GpuConvolveSeparableCols(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block_grid_width = (int)ceil((float)image_width * 3 / (COL_TILE_WIDTH * COL_HTILES_IN_BLOCK));
    int block_grid_height = (int)ceil((float)image_height / (COL_VTILES_IN_BLOCK * COL_TILE_HEIGHT));
    printf("block_grid_width:%d block_grid_height:%d\n", block_grid_width, block_grid_height);
    printf("image_width:%d image_height:%d\n", image_width, image_height);
    dim3 blocks(block_grid_width, block_grid_height);
    dim3 threads(COL_TILE_WIDTH, COL_TILE_HEIGHT);
    cudaEventRecord(start, 0);
    convolveSeparableColsKernel << <blocks, threads >> > (
        d_dst,
        d_src,
        d_depth_map,
        image_width,
        image_height,
        pitch,
        depth_map_pitch,
        focus_depth
        );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


void depthProcess(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, double fx, double bl, float scale, int doffs)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    depth << <grid, block >> > (src, dst, dst.rows, dst.cols, fx, bl, scale, doffs);

}


void normal(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int dimX, int dimY, int step, float min, float max)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    k_normalize << <grid, block >> > (src, dst, dst.rows, dst.cols, step, min, max);
}

/*
__global__ void horizontal(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, const cv::cuda::PtrStepSzf dm,
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, float focus_depth)
{

    __shared__ uchar3 temp[BLOCK_SIZE][BLOCK_SIZE + 2 * RADIUS];

    // local indices
    int lindex_X = threadIdx.x + RADIUS;
    int lindex_Y = threadIdx.y;

    int dst_x = blockDim.x * blockIdx.x + lindex_X;
    int dst_y = blockDim.y * blockIdx.y + lindex_Y;

    if (dst_x < cols && dst_y < rows)
    {

        // Read input elements into shared memory
        temp[lindex_Y][lindex_X] = src(dst_y, dst_x);
        if (threadIdx.x < RADIUS) {
            temp[lindex_Y][lindex_X - RADIUS] = src(dst_y, dst_x - RADIUS);
            if (dst_x + BLOCK_SIZE < cols)
                temp[lindex_Y][lindex_X + BLOCK_SIZE] = src(dst_y, dst_x + BLOCK_SIZE);
        }
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (dst_x < cols - RADIUS && dst_y < rows && dst_x > RADIUS && dst_y > 0)
    {

        // Apply the kernel
        float tmp[3] = { 0,0,0 };
        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            //int kernel_radius = floorf((KERNEL_RADIUS)*fabs(dm(lindex_Y, lindex_X) - focus_depth));
            //int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;

            //tmp[0] += (float)(temp[lindex_Y][lindex_X + i].x) * d_kernelGaussConv(0, i + RADIUS);
            //tmp[1] += (float)(temp[lindex_Y][lindex_X + i].y) * d_kernelGaussConv(0, i + RADIUS);
            //tmp[2] += (float)(temp[lindex_Y][lindex_X + i].z) * d_kernelGaussConv(0, i + RADIUS);
            //if (kernel_radius > 0) {

                tmp[0] += (float)(temp[lindex_Y][lindex_X + i].x) * c_kernel[ i+RADIUS];
                tmp[1] += (float)(temp[lindex_Y][lindex_X + i].y) * c_kernel[ i+RADIUS];
                tmp[2] += (float)(temp[lindex_Y][lindex_X + i].z) * c_kernel[ i+RADIUS];
            //}
            /*
            else {
                tmp[0] += (float)(temp[lindex_Y][lindex_X + i].x);
                tmp[1] += (float)(temp[lindex_Y][lindex_X + i].y);
                tmp[2] += (float)(temp[lindex_Y][lindex_X + i].z);

            }



            if (dm(dst_y, dst_x) < 0.3 || dm(dst_y, dst_x) == 0)
            {
                dst(dst_y, dst_x).x = (unsigned char)(tmp[0]);
                dst(dst_y, dst_x).y = (unsigned char)(tmp[1]);
                dst(dst_y, dst_x).z = (unsigned char)(tmp[2]);
            }

            else
            {
                dst(dst_y, dst_x).x = (unsigned char)(src(dst_y, dst_x).x);
                dst(dst_y, dst_x).y = (unsigned char)(src(dst_y, dst_x).y);
                dst(dst_y, dst_x).z = (unsigned char)(src(dst_y, dst_x).z);
            }
        }
    }

}


__global__ void vertical(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, const cv::cuda::PtrStepSzf dm,
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, float focus_depth)
{

    __shared__ uchar3 temp2[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE];

    // local indices
    int lindex_X2 = threadIdx.x;
    int lindex_Y2 = threadIdx.y + RADIUS;

    int dst_x2 = blockDim.x * blockIdx.x + lindex_X2;
    int dst_y2 = blockDim.y * blockIdx.y + lindex_Y2;

    if (dst_x2 < cols && dst_y2 < rows - RADIUS && dst_x2 > 0 && dst_y2 > RADIUS)
    {

        // Read input elements into shared memory
        temp2[lindex_Y2][lindex_X2] = src(dst_y2, dst_x2);
        if (threadIdx.y < RADIUS) {
            temp2[lindex_Y2 - RADIUS][lindex_X2] = src(dst_y2 - RADIUS, dst_x2);
            if (dst_y2 + BLOCK_SIZE < rows)
                temp2[lindex_Y2 + BLOCK_SIZE][lindex_X2] = src(dst_y2 + BLOCK_SIZE, dst_x2);
        }
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (dst_x2 < cols && dst_y2 < rows)
    {
        
            // Apply the kernel
            float tmp2[3] = { 0,0,0 };
            
           

            for (int i = -RADIUS; i <= RADIUS; i++)
            {
                //int kernel_radius = floorf((KERNEL_RADIUS)*fabs(dm(0,0) - focus_depth));
                //int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;
                
                //tmp2[0] += (float)(temp2[lindex_Y2 + i][lindex_X2].x) * d_kernelGaussConv(0, i + RADIUS);
                //tmp2[1] += (float)(temp2[lindex_Y2 + i][lindex_X2].y) * d_kernelGaussConv(0, i + RADIUS);
                //tmp2[2] += (float)(temp2[lindex_Y2 + i][lindex_X2].z) * d_kernelGaussConv(0, i + RADIUS);
                if (RADIUS > 0) {
                    
                    tmp2[0] += (float)(temp2[lindex_Y2][lindex_X2 + i].x) * c_kernel[ i+RADIUS];
                    tmp2[1] += (float)(temp2[lindex_Y2][lindex_X2 + i].y) * c_kernel[ i+RADIUS];
                    tmp2[2] += (float)(temp2[lindex_Y2][lindex_X2 + i].z) * c_kernel[ i+RADIUS];
                    
                }

                //else {
                //    tmp2[0] += (float)(temp2[lindex_Y2][lindex_X2 + i].x);
                //    tmp2[1] += (float)(temp2[lindex_Y2][lindex_X2 + i].y);
                //    tmp2[2] += (float)(temp2[lindex_Y2][lindex_X2 + i].z);
                //}

                dst(dst_y2, dst_x2).x =  tmp2[0];
                dst(dst_y2, dst_x2).y =  tmp2[1];
                dst(dst_y2, dst_x2).z =  tmp2[2];
                
            }



            if (dm(dst_y2, dst_x2) < 0.3 || dm(dst_y2, dst_x2) == 0)
            {
            dst(dst_y2, dst_x2).x = (unsigned char)(tmp2[0]);
            dst(dst_y2, dst_x2).y = (unsigned char)(tmp2[1]);
            dst(dst_y2, dst_x2).z = (unsigned char)(tmp2[2]);
            }
            else
            {
                dst(dst_y2, dst_x2).x = (unsigned char)(src(dst_y2, dst_x2).x);
                dst(dst_y2, dst_x2).y = (unsigned char)(src(dst_y2, dst_x2).y);
                dst(dst_y2, dst_x2).z = (unsigned char)(src(dst_y2, dst_x2).z);
            }
        
    }

}



__global__ void gaussian(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, const cv::cuda::PtrStepSzf dm, int rows, int cols,
    cv::cuda::PtrStep<float> d_gaussConv, int kernelSize, int sigma)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const float rad = (kernelSize - 1.0) / 2.0;

    if (dst_x < cols - rad && dst_y < rows - rad &&
        dst_x > rad && dst_y > rad) {
        float tmp[3] = { 0.0, 0.0, 0.0, };

        for (int i = dst_x - rad; i <= dst_x + rad; i++) {
            for (int j = dst_y - rad; j <= dst_y + rad; j++) {

                    tmp[0] += (float)src(j, i).x * d_gaussConv(j - dst_y + rad, i - dst_x + rad);

                    tmp[1] += (float)src(j, i).y * d_gaussConv(j - dst_y + rad, i - dst_x + rad);

                    tmp[2] += (float)src(j, i).z * d_gaussConv(j - dst_y + rad, i - dst_x + rad);
            }
        }

        if (dm(dst_y, dst_x) < 0.3 || dm(dst_y, dst_x) == 0)
        {
            dst(dst_y, dst_x).x = (unsigned char)(tmp[0]);
            dst(dst_y, dst_x).y = (unsigned char)(tmp[1]);
            dst(dst_y, dst_x).z = (unsigned char)(tmp[2]);
        }
        else
        {
            dst(dst_y, dst_x).x = (unsigned char)(src(dst_y, dst_x).x);
            dst(dst_y, dst_x).y = (unsigned char)(src(dst_y, dst_x).y);
            dst(dst_y, dst_x).z = (unsigned char)(src(dst_y, dst_x).z);
        }

    }

}




void gaussianConv(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dm, int dimX, int dimY, cv::cuda::GpuMat& d_gaussConv, int kernelSize, int sigma)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussian <<<grid, block >> > (src, dst, dm, dst.rows, dst.cols, d_gaussConv, kernelSize, sigma);

}



void gaussianSep(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dm, cv::cuda::GpuMat& d_kernelGauss, int pass, float focus_depth)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    if (pass == 1)
    {
        horizontal << <grid, block >> > (src, dst, dm, dst.rows, dst.cols, d_kernelGauss,  focus_depth);
    }
    else
    {
        vertical << <grid, block >> > (src, dst, dm, dst.rows, dst.cols, d_kernelGauss,  focus_depth);
    }

}*/