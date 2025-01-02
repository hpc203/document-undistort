import cv2
from binary_predictor import UnetCNN
from unblur_predictor import NAF_DPM, OpenCvBilateral
from unshadow_predictor import GCDRNET
from unwrap_predictor import UVDocPredictor


if __name__=='__main__':
    binary_model = UnetCNN('weights/unetcnn.onnx')
    unblur_model = NAF_DPM('weights/nafdpm.onnx')
    unblur_model2 = OpenCvBilateral()
    unshadow_model = GCDRNET('weights/gcnet.onnx', 'weights/drnet.onnx')
    unwrap_model = UVDocPredictor('weights/uvdoc.onnx')
    model_dict = {"binary": binary_model, "unblur": unblur_model, "unshadow": unshadow_model, "unwrap": unwrap_model, "OpenCvBilateral": unblur_model2}

    task_list = ["unwrap", "unshadow", "unblur", "OpenCvBilateral"]
    srcimg = cv2.imread('images/demo3.jpg')
    out_img = srcimg.copy()
    for task in task_list:
        out_img = model_dict[task].predict(out_img)
        
    cv2.imwrite('out.jpg', out_img)
    