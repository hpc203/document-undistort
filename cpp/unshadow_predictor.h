#ifndef UNSHADOW_PREDICTOR_H
#define UNSHADOW_PREDICTOR_H
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>


class GCDRNET
{
public:
	GCDRNET(const std::string& gcnet_modelpath, const std::string& drnet_modelpath);
	cv::Mat predict(const cv::Mat& srcimg);   
private:
	void preprocess(const cv::Mat& img);
	std::vector<float> input_image;
    std::vector<float> concatenated_input;
    int input_h;
    int input_w;
    int padding_h;
    int padding_w;
    cv::Mat stride_integral(const cv::Mat& img, const int stride);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "UNSHADOW PREDICTOR");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *gcnet_session = nullptr;
    Ort::Session *drnet_session = nullptr;
	const std::vector<const char*> input_names = {"input"};
	const std::vector<const char*> output_names = {"output"};

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;
};


#endif