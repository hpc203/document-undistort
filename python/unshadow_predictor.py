import cv2
import numpy as np
import onnxruntime


class GCDRNET():
    def __init__(self, gcnet_modelpath, drnet_modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.gcnet_session = onnxruntime.InferenceSession(gcnet_modelpath, so)
        self.drnet_session = onnxruntime.InferenceSession(drnet_modelpath, so)
        self.gcnet_input_name = self.gcnet_session.get_inputs()[0].name
        self.drnet_input_name = self.drnet_session.get_inputs()[0].name

    def stride_integral(self, img, stride=32):
        h, w = img.shape[:2]

        if (h % stride) != 0:
            padding_h = stride - (h % stride)
            img = cv2.copyMakeBorder(img, padding_h, 0, 0, 0, borderType=cv2.BORDER_REPLICATE)
        else:
            padding_h = 0

        if (w % stride) != 0:
            padding_w = stride - (w % stride)
            img = cv2.copyMakeBorder(img, 0, 0, padding_w, 0, borderType=cv2.BORDER_REPLICATE)
        else:
            padding_w = 0

        return img, padding_h, padding_w

    def preprocess(self, img):
        img, padding_h, padding_w = self.stride_integral(img)
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img, padding_h, padding_w

    def predict(self, img):
        im_padding, padding_h, padding_w = self.preprocess(img.copy())
        img_shadow = im_padding.copy()
        img_shadow = self.gcnet_session.run(None, {self.gcnet_input_name: img_shadow})[0]
        model1_im = np.clip(im_padding / img_shadow, 0, 1)
        # 拼接 im_org 和 model1_im
        concatenated_input = np.concatenate((im_padding, model1_im), axis=1)
        pred = self.drnet_session.run(None, {self.drnet_input_name: concatenated_input})[0]
        out_img = self.postprocess(pred, padding_h, padding_w)
        return out_img.astype(np.uint8)
    
    def postprocess(self, pred, padding_h, padding_w):
        pred = np.transpose(pred[0], (1, 2, 0))
        pred = pred * 255
        enhance_img = pred[padding_h:, padding_w:]
        return enhance_img


if __name__=='__main__':
    model = GCDRNET('weights/gcnet.onnx', 'weights/drnet.onnx')
    img = cv2.imread('images/demo3.jpg')
    out_img = model.predict(img)
    cv2.imwrite('unshadow_predictor_out.jpg', out_img)