import cv2
import numpy as np
import onnxruntime


def restore_original_size(image, pad_info):
    start_x, start_y, original_height, original_width = pad_info
    cropped_image = image[start_y:start_y + original_height, start_x:start_x + original_width]
    return cropped_image

def pad_to_multiple_of_n(image, n=32):
    original_height, original_width = image.shape[:2]

    # 计算目标形状
    target_width = ((original_width + n - 1) // n) * n
    target_height = ((original_height + n - 1) // n) * n

    # 创建一个纯白背景的图像
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # 计算填充的位置
    start_x = (target_width - original_width) // 2
    start_y = (target_height - original_height) // 2

    # 将原始图像放置在纯白背景上
    padded_image[start_y:start_y + original_height, start_x:start_x + original_width] = image

    # 返回填充后的图像和填充位置
    return padded_image, (start_x, start_y, original_height, original_width)


class UnetCNN():
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(modelpath, so)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad_info = pad_to_multiple_of_n(img)
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        # 将图像数据扩展为一个批次的形式
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img, pad_info
    
    def predict(self, img):
        img, pad_info = self.preprocess(img)
        pred = self.session.run(None, {self.input_name: img})[0]
        out_img = self.postprocess(pred, pad_info)
        return out_img.astype(np.uint8)

    def postprocess(self, img, pad_info):
        img = 1 - (img - img.min()) / (img.max() - img.min())
        img = img[0].transpose(1, 2, 0)
        # 重复最后一个通道维度三次
        img = np.repeat(img, 3, axis=2)
        img = (img * 255 + 0.5).clip(0, 255)
        img = restore_original_size(img, pad_info)
        return img


if __name__=='__main__':
    model = UnetCNN('weights/unetcnn.onnx')
    img = cv2.imread('images/demo3.jpg')
    out_img = model.predict(img)
    cv2.imwrite('binary_predictor_out.jpg', out_img)