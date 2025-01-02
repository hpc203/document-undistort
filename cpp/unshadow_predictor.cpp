#include "unshadow_predictor.h"


using namespace cv;
using namespace std;
using namespace Ort;


GCDRNET::GCDRNET(const string& gcnet_modelpath, const string& drnet_modelpath)
{
    if (gcnet_modelpath.empty() || drnet_modelpath.empty()) 
    {
        std::cout << "onnx path error" << std::endl;
    }

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    // 加载模型
    std::wstring_convert <std::codecvt_utf8<wchar_t>> converter;
#ifdef _WIN32
    std::wstring w_model_path = converter.from_bytes(gcnet_modelpath);
    gcnet_session = new Ort::Session(env, w_model_path.c_str(), sessionOptions);
    std::wstring w_model2_path = converter.from_bytes(drnet_modelpath);
    drnet_session = new Ort::Session(env, w_model2_path.c_str(), sessionOptions);
#else
    gcnet_session = new Ort::Session(env, gcnet_modelpath.c_str(), sessionOptions);
    drnet_session = new Ort::Session(env, drnet_modelpath.c_str(), sessionOptions);
#endif
}

Mat GCDRNET::stride_integral(const Mat& srcimg, const int stride)
{
    int h = srcimg.rows;
    int w = srcimg.cols;

    Mat img = srcimg.clone();
    if((h % stride) != 0)
    {
        this->padding_h = stride - (h % stride);
        cv::copyMakeBorder(img, img, padding_h, 0, 0, 0, cv::BORDER_REPLICATE);
    }
    else
    {
        this->padding_h = 0;
    }

    if((w % stride) != 0)
    {
        this->padding_w = stride - (w % stride);
        cv::copyMakeBorder(img, img, 0, 0, padding_w, 0, cv::BORDER_REPLICATE);
    }
    else
    {
        this->padding_w = 0;
    }
    return img;
}

void GCDRNET::preprocess(const Mat& srcimg)
{
    Mat img = this->stride_integral(srcimg, 32);
    this->input_h = img.rows;
    this->input_w = img.cols;
    vector<cv::Mat> bgrChannels(3);
    split(img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.0);
    }

    const int image_area = this->input_h * this->input_w;
    this->input_image.clear();
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

Mat GCDRNET::predict(const Mat& srcimg)
{
    this->preprocess(srcimg);
    std::vector<int64_t> input_img_shape = {1, 3, this->input_h, this->input_w};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> gcnet_outputs = this->gcnet_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    float* img_shadow = gcnet_outputs[0].GetTensorMutableData<float>();
    const int len = 3 * this->input_h * this->input_w;
    this->concatenated_input.clear();
    this->concatenated_input.resize(2 * len);
    for(int i=0;i<len;i++)
    {
        this->concatenated_input[i] = this->input_image[i];
        float x = this->input_image[i] / img_shadow[i];
        x = std::min(std::max(x, 0.0f), 1.0f);
        this->concatenated_input[i + len] = x;
    }

    std::vector<int64_t> input2_shape = {1, 2*3, this->input_h, this->input_w};
    Value input2_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->concatenated_input.data(), this->concatenated_input.size(), input2_shape.data(), input2_shape.size());

    vector<Value> drnet_outputs = this->drnet_session->Run(runOptions, this->input_names.data(), &input2_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());

    std::vector<int64_t> out_shape = drnet_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_h = out_shape[2];
    const int out_w = out_shape[3];
    const int area = out_h * out_w;
    float* pred = drnet_outputs[0].GetTensorMutableData<float>();   
    Mat bmat = Mat(out_h, out_w, CV_32FC1, pred);
    Mat gmat = Mat(out_h, out_w, CV_32FC1, pred + area);
    Mat rmat = Mat(out_h, out_w, CV_32FC1, pred + area * 2);
    bmat *= 255;
    gmat *= 255;
    rmat *= 255;

    vector<Mat> channel_mats = {bmat, gmat, rmat};
    Mat out_img;
    merge(channel_mats, out_img);
    Mat enhance_img;
    out_img(Rect(this->padding_w, this->padding_h, out_w-this->padding_w, out_h-this->padding_h)).copyTo(enhance_img);
    enhance_img.convertTo(enhance_img, CV_8UC3);
    return enhance_img;
}