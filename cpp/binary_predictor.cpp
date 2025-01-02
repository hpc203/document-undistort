#include "binary_predictor.h"


using namespace cv;
using namespace std;
using namespace Ort;


static Mat pad_to_multiple_of_n(const Mat& image, const int n, int* pad_info)
{
    int original_height = image.rows;
    int original_width = image.cols;
    
    int target_width = ((original_width + n - 1) / n) * n;
    int target_height = ((original_height + n - 1) / n) * n;

    Mat padded_image(target_height, target_width, CV_8UC3, Scalar(255,255,255));

    int start_x = (target_width - original_width) / 2;
    int start_y = (target_height - original_height) / 2;

    image.copyTo(padded_image(Rect(start_x, start_y, original_width, original_height)));
    pad_info[0] = start_x;
    pad_info[1] = start_y;
    pad_info[2] = original_height;
    pad_info[3] = original_width;
    return padded_image;
}

UnetCNN::UnetCNN(const string& model_path)
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

void UnetCNN::preprocess(const Mat& srcimg)
{
    Mat img = pad_to_multiple_of_n(srcimg, 32, this->pad_info);
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
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
}

Mat UnetCNN::predict(const Mat& srcimg)
{
    this->preprocess(srcimg);
    std::vector<int64_t> input_img_shape = {1, 3, this->input_h, this->input_w};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());

    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_h = out_shape[2];
    const int out_w = out_shape[3];
    float* pred = ort_outputs[0].GetTensorMutableData<float>();   
    Mat out = Mat(out_h, out_w, CV_32FC1, pred);
    Mat cropped_image = this->postprocess(out);
    out.release();
    vector<Mat> channel_mats = {cropped_image, cropped_image, cropped_image};
    Mat out_img;
    merge(channel_mats, out_img);
    out_img.convertTo(out_img, CV_8UC3);
    return out_img;
}

Mat UnetCNN::postprocess(Mat& img)
{
    double min_value, max_value;
	minMaxLoc(img, &min_value, &max_value, 0, 0);
    img = 1 - (img - min_value) / (max_value - min_value);
    img = img * 255 + 0.5;
    img.setTo(0, img<0);
    img.setTo(255, img>255);
    Mat cropped_image;
    img(Rect(this->pad_info[0], this->pad_info[1], this->pad_info[3], this->pad_info[2])).copyTo(cropped_image);
    return cropped_image;
}