import cv2
import numpy as np
import onnxruntime


class NAF_DPM():
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(modelpath, so)
        self.input_name = self.session.get_inputs()[0].name
    
    def preprocess(self, img):
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        # 将图像数据扩展为一个批次的形式
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img
    
    def predict(self, img):
        img = self.preprocess(img)
        pred = self.session.run(None, {self.input_name: img})[0]
        out_img = self.postprocess(pred)
        return out_img.astype(np.uint8)
    
    def postprocess(self, img):
        img = img[0]
        img = (img * 255 + 0.5).clip(0, 255).transpose(1, 2, 0)
        return img


class OpenCvBilateral:
    def __init__(self,):
        pass
    def predict(self, img):
        img = img.astype(np.uint8)
        # 双边滤波
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        # 自适应直方图均衡化
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 应用锐化滤波器
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened


if __name__=='__main__':
    model = NAF_DPM('weights/nafdpm.onnx')
    model2 = OpenCvBilateral()
    img = cv2.imread('images/demo3.jpg')
    out_img = model.predict(img)
    out_img = model2.predict(out_img)
    cv2.imwrite('unblur_predictor_out.jpg', out_img)