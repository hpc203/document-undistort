#include "unwrap_predictor.h"


using namespace cv;
using namespace std;
using namespace Ort;


UVDocPredictor::UVDocPredictor(const string& model_path)
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

void UVDocPredictor::preprocess(Mat& img)
{
    img.convertTo(img, CV_32FC3, 1 / 255.0);

    Mat temp;
    cv::resize(img, temp, cv::Size(this->input_w, this->input_h));
    vector<cv::Mat> bgrChannels(3);
    split(temp, bgrChannels);
    const int image_area = this->input_h * this->input_w;
    this->input_image.clear();
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

Mat UVDocPredictor::predict(const Mat& srcimg)
{
    const int size[2] = {srcimg.cols, srcimg.rows};
    Mat img = srcimg.clone();
    this->preprocess(img);
    std::vector<int64_t> input_img_shape = {1, 3, this->input_h, this->input_w};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());

    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* output = ort_outputs[0].GetTensorMutableData<float>();
    Mat out_img = this->postprocess(img, size, output, out_shape);
    out_img.convertTo(out_img, CV_8UC3);
    return out_img;
}

static Mat convert3channeltonchw(const Mat& img)
{
    vector<cv::Mat> bgrChannels(3);
    split(img, bgrChannels);
    const int image_area = img.rows * img.cols;
    const vector<int> newsz = {1, 3, img.rows, img.cols};
    Mat dstimg = cv::Mat(newsz, CV_32FC1);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy((float *)dstimg.data, (float *)bgrChannels[0].data, single_chn_size);
    memcpy((float *)dstimg.data + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy((float *)dstimg.data + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
    return dstimg;
}

Mat UVDocPredictor::postprocess(const Mat& img, const int* size, const float* output, vector<int64_t> out_shape)
{
    Mat warped_img = convert3channeltonchw(img);

    Mat upsampled_grid = this->interpolate(output, out_shape, size, true);
    cv::transposeND(upsampled_grid, {0, 2, 3, 1}, upsampled_grid);
    
    Mat unwarped_img = this->grid_sample(warped_img, upsampled_grid, true);

    float* pdata = (float*)unwarped_img.data;
    const int out_h = unwarped_img.size[2];
    const int out_w = unwarped_img.size[3];
    const int area = out_h * out_w;
    Mat bmat = Mat(out_h, out_w, CV_32FC1, pdata);
    Mat gmat = Mat(out_h, out_w, CV_32FC1, pdata + area);
    Mat rmat = Mat(out_h, out_w, CV_32FC1, pdata + area * 2);
    bmat *= 255;
    gmat *= 255;
    rmat *= 255;
    vector<Mat> channel_mats = {bmat, gmat, rmat};
    Mat out_img;
    merge(channel_mats, out_img);
    return out_img;
}

static float get_pixel_value(const float* pinput, const int H, const int W, const int y, const int x)
{
    if(y < 0 || y >= H || x < 0 || x >= W)
    {
        return 0.f;
    }
    return pinput[y*W+x];
}

Mat UVDocPredictor::interpolate(const float* input_tensor, vector<int64_t> shape, const int* size, const bool align_corners)
{
    const int B = shape[0];
    const int C = shape[1];
    const int H = shape[2];
    const int W = shape[3];
    const int new_H = size[1];
    const int new_W = size[0];
    const vector<int> newsz = {B, C, new_H, new_W};
    Mat dstimg = cv::Mat(newsz, CV_32FC1);
    for(int n=0;n<B;n++)
    {
        for(int cid=0;cid<C;cid++)
        {
            float scale_h = (new_H > 1) ? (float(H - 1) / float(new_H - 1)):0.f;
            float scale_w = (new_W > 1) ? (float(W - 1) / float(new_W - 1)):0.f;
            if(!align_corners)
            {
                scale_h = (float)H / new_H;
                scale_w = (float)W / new_W;
            }
            const float* pinput = input_tensor + n*C*H*W + cid*H*W;
            for(int h=0;h<new_H;h++)
            {
                for(int w=0;w<new_W;w++)
                {
                    const float y = (float)h * scale_h;
                    const float x = (float)w * scale_w;
                    const int y0 = floor(y);
                    const int x0 = floor(x);
                    const int y1 = y0 + 1;
                    const int x1 = x0 + 1;
                    const float denom = (y1-y0)*(x1-x0);

                    const float f_x0_y0 = get_pixel_value(pinput, H, W, y0, x0);
                    const float f_x1_y0 = get_pixel_value(pinput, H, W, y0, x1);
                    const float f_x0_y1 = get_pixel_value(pinput, H, W, y1, x0);
                    const float f_x1_y1 = get_pixel_value(pinput, H, W, y1, x1);

                    const float f = ((y1-y)*(x1-x) / denom) * f_x0_y0 + ((y1-y)*(x-x0) / denom) * f_x1_y0 + ((y-y0)*(x1-x) / denom) * f_x0_y1 + ((y-y0)*(x-x0) / denom) * f_x1_y1;
                    dstimg.ptr<float>(n, cid, h)[w] = f;
                }
            }
        }
    }
    return dstimg;
}

Mat UVDocPredictor::grid_sample(const Mat& input_tensor, const Mat& grid, const bool align_corners)
{
    const int B = input_tensor.size[0];
    const int C = input_tensor.size[1];
    const int H = input_tensor.size[2];
    const int W = input_tensor.size[3];
    const int B_grid = grid.size[0];
    const int H_grid = grid.size[1];
    const int W_grid = grid.size[2];

    if(B != B_grid || H != H_grid || W != W_grid)
    {
        cout<<"Error, Input tensor and grid must have the same spatial dimensions."<<endl;
        exit(-1);
    }

    const vector<int> newsz = {B, C, H, W};
    Mat dstimg = cv::Mat(newsz, CV_32FC1);
    for(int n=0;n<B;n++)
    {
        for(int cid=0;cid<C;cid++)
        {
            const float* pinput = (float*)input_tensor.data + n*C*H*W + cid*H*W;
            for(int h=0;h<H_grid;h++)
            {
                for(int w=0;w<W_grid;w++)
                {
                    float x = (grid.ptr<float>(n, h, w)[0] + 1) * (W - 1) / 2;
                    float y = (grid.ptr<float>(n, h, w)[1] + 1) * (H - 1) / 2;
                    if(!align_corners)
                    {
                        x = ((grid.ptr<float>(n, h, w)[0] + 1) * W - 1) / 2;
                        y = ((grid.ptr<float>(n, h, w)[1] + 1) * H - 1) / 2;
                    }
                    
                    const int y0 = floor(y);
                    const int x0 = floor(x);
                    const int y1 = y0 + 1;
                    const int x1 = x0 + 1;
                    const float denom = (y1-y0)*(x1-x0);

                    const float f_x0_y0 = get_pixel_value(pinput, H, W, y0, x0);
                    const float f_x1_y0 = get_pixel_value(pinput, H, W, y0, x1);
                    const float f_x0_y1 = get_pixel_value(pinput, H, W, y1, x0);
                    const float f_x1_y1 = get_pixel_value(pinput, H, W, y1, x1);

                    const float f = ((y1-y)*(x1-x) / denom) * f_x0_y0 + ((y1-y)*(x-x0) / denom) * f_x1_y0 + ((y-y0)*(x1-x) / denom) * f_x0_y1 + ((y-y0)*(x-x0) / denom) * f_x1_y1;
                    dstimg.ptr<float>(n, cid, h)[w] = f;
                }
            }
        }
    }
    return dstimg;
}