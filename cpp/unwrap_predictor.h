#ifndef UNWRAP_PREDICTOR_H
#define UNWRAP_PREDICTOR_H
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>


class UVDocPredictor
{
public:
	UVDocPredictor(const std::string& model_path);
	cv::Mat predict(const cv::Mat& srcimg);   
private:
	void preprocess(cv::Mat& img);
	std::vector<float> input_image;
    const int input_h = 712;
    const int input_w = 488;
    const int grid_size[2] = {45, 31};
    cv::Mat postprocess(const cv::Mat& img, const int* size, const float* output, std::vector<int64_t> out_shape);
    cv::Mat interpolate(const float* input_tensor, std::vector<int64_t> shape, const int* size, const bool align_corners);
    cv::Mat grid_sample(const cv::Mat& input_tensor, const cv::Mat& grid, const bool align_corners);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "UNWRAP PREDICTOR");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *ort_session = nullptr;
	const std::vector<const char*> input_names = {"input"};
	const std::vector<const char*> output_names = {"output", "546"};

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;
};


#endif