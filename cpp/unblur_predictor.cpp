#include "unblur_predictor.h"


using namespace cv;
using namespace std;
using namespace Ort;


NAF_DPM::NAF_DPM(const string& model_path)
{
    if (model_path.empty()) 
    {
        std::cout << "onnx path error" << std::endl;
    }

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    // 加载模型
    std::wstring_convert <std::codecvt_utf8<wchar_t>> converter;
#ifdef _WIN32
    std::wstring w_model_path = converter.from_bytes(model_path);
    ort_session = new Ort::Session(env, w_model_path.c_str(), sessionOptions);
#else
    ort_session = new Ort::Session(env, model_path.c_str(), sessionOptions);
#endif
}

void NAF_DPM::preprocess(const Mat& srcimg)
{
    this->input_h = srcimg.rows;
    this->input_w = srcimg.cols;
    vector<cv::Mat> bgrChannels(3);
    split(srcimg, bgrChannels);
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

Mat NAF_DPM::predict(const Mat& srcimg)
{
    this->preprocess(srcimg);
    std::vector<int64_t> input_img_shape = {1, 3, this->input_h, this->input_w};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());

    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_h = out_shape[2];
    const int out_w = out_shape[3];
    const int area = out_h * out_w;
    float* pred = ort_outputs[0].GetTensorMutableData<float>();   
    Mat bmat = Mat(out_h, out_w, CV_32FC1, pred);
    Mat gmat = Mat(out_h, out_w, CV_32FC1, pred + area);
    Mat rmat = Mat(out_h, out_w, CV_32FC1, pred + area * 2);

    this->postprocess(bmat);
    this->postprocess(gmat);
    this->postprocess(rmat);

    vector<Mat> channel_mats = {bmat, gmat, rmat};
    Mat out_img;
    merge(channel_mats, out_img);
    out_img.convertTo(out_img, CV_8UC3);
    return out_img;
}

void NAF_DPM::postprocess(Mat& img)
{
    img = img * 255 + 0.5;
    img.setTo(0, img<0);
    img.setTo(255, img>255);
}


cv::Mat OpenCvBilateral(const cv::Mat& img) {
    cv::Mat img_uint8;
    img.convertTo(img_uint8, CV_8U);

    // 双边滤波
    cv::Mat bilateral;
    cv::bilateralFilter(img_uint8, bilateral, 9, 75, 75);

    // 自适应直方图均衡化
    cv::Mat lab;
    cv::cvtColor(bilateral, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab);
    cv::Mat enhanced;
    cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);

    // 应用锐化滤波器
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::Mat sharpened;
    cv::filter2D(enhanced, sharpened, -1, kernel);

    return sharpened;
}