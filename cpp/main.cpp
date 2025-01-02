#include "binary_predictor.h"
#include "unblur_predictor.h"
#include "unshadow_predictor.h"
#include "unwrap_predictor.h"
#include<opencv2/highgui.hpp>


using namespace cv;
using namespace std;


int main()
{
    UnetCNN binary_model("/home/wangbo/doc-undistort/weights/unetcnn.onnx");   /////注意文件路径要写对
    NAF_DPM unblur_model("/home/wangbo/doc-undistort/weights/nafdpm.onnx");
    GCDRNET unshadow_model("/home/wangbo/doc-undistort/weights/gcnet.onnx", "/home/wangbo/doc-undistort/weights/drnet.onnx");
    UVDocPredictor unwrap_model("/home/wangbo/doc-undistort/weights/uvdoc.onnx");

    vector<string> task_list = {"unwrap", "unshadow", "unblur", "OpenCvBilateral"};

    string imgpath = "/home/wangbo/doc-undistort/images/demo3.jpg";

    Mat srcimg = imread(imgpath);
    // Mat out_img = unwrap_model.predict(srcimg);

    Mat out_img = srcimg.clone();
    for(string task : task_list)
    {
        /////switch不支持字符串表达式,所以用if else
        if(task == "unwrap")
        {
            out_img = unwrap_model.predict(out_img);
        }
        else if(task == "unshadow")
        {
            out_img = unshadow_model.predict(out_img);
        }
        else if(task == "unblur")
        {
            out_img = unblur_model.predict(out_img);
        }
        else if(task == "OpenCvBilateral")
        {
            out_img = OpenCvBilateral(out_img);
        }
        else if(task == "binary")
        {
            out_img = binary_model.predict(out_img);
        }
        else
        {
            cout << "task not found" << endl;
        }
        
    }

    imwrite("out.jpg", out_img);
    return 0;
}
