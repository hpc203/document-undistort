import cv2
import numpy as np
import onnxruntime
from scipy.ndimage import map_coordinates


class UVDocPredictor:
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(modelpath, so)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = [488, 712]
        self.grid_size = [45, 31]

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        size = img.shape[:2][::-1]
        img = img.astype(np.float32) / 255
        inp = self.preprocess(img.copy())
        outputs = self.session.run(None, {self.input_name: inp})[0]
        out_img = self.postprocess(img, size, outputs)
        return out_img.astype(np.uint8)
    
    def postprocess(self, img, size, output):
        # 将图像转换为NumPy数组
        warped_img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)

        # 上采样网格
        upsampled_grid = self.interpolate(output, size=(size[1], size[0]), align_corners=True)
        # 调整网格的形状
        upsampled_grid = upsampled_grid.transpose(0, 2, 3, 1)

        # 重映射图像
        unwarped_img = self.grid_sample(warped_img, upsampled_grid)

        # 将结果转换回原始格式
        return unwarped_img[0].transpose(1, 2, 0) * 255

    def interpolate(self, input_tensor, size, align_corners=True):
        """
        Interpolate function to resize the input tensor.

        Args:
            input_tensor: numpy.ndarray of shape (B, C, H, W)
            size: tuple of int (new_height, new_width)
            mode: str, interpolation mode ('bilinear' or 'nearest')
            align_corners: bool, whether to align corners in bilinear interpolation

        Returns:
            numpy.ndarray of shape (B, C, new_height, new_width)
        """
        B, C, H, W = input_tensor.shape
        new_H, new_W = size
        resized_tensors = []
        for b in range(B):
            resized_channels = []
            for c in range(C):
                # 计算新的坐标
                if align_corners:
                    scale_h = (H - 1) / (new_H - 1) if new_H > 1 else 0
                    scale_w = (W - 1) / (new_W - 1) if new_W > 1 else 0
                else:
                    scale_h = H / new_H
                    scale_w = W / new_W

                # 创建新的坐标网格
                y, x = np.indices((new_H, new_W), dtype=np.float32)
                y = y * scale_h
                x = x * scale_w

                # 双线性插值
                coords = np.stack([y.flatten(), x.flatten()], axis=0)
                # tmp = my_map_coordinates(input_tensor[b, c], coords)
                resized_channel = map_coordinates(input_tensor[b, c], coords, order=1, mode='constant', cval=0.0)
                resized_channel = resized_channel.reshape(new_H, new_W)
                resized_channels.append(resized_channel)

            resized_tensors.append(np.stack(resized_channels, axis=0))

        return np.stack(resized_tensors, axis=0)

    def grid_sample(self, input_tensor, grid, align_corners=True):
        """
        Grid sample function to sample the input tensor using the given grid.

        Args:
            input_tensor: numpy.ndarray of shape (B, C, H, W)
            grid: numpy.ndarray of shape (B, H, W, 2) with values in [-1, 1]
            align_corners: bool, whether to align corners in bilinear interpolation

        Returns:
            numpy.ndarray of shape (B, C, H, W)
        """
        B, C, H, W = input_tensor.shape
        B_grid, H_grid, W_grid, _ = grid.shape

        if B != B_grid or H != H_grid or W != W_grid:
            raise ValueError("Input tensor and grid must have the same spatial dimensions.")

        # Convert grid coordinates from [-1, 1] to [0, W-1] and [0, H-1]
        if align_corners:
            grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) * (W - 1) / 2
            grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) * (H - 1) / 2
        else:
            grid[:, :, :, 0] = ((grid[:, :, :, 0] + 1) * W - 1) / 2
            grid[:, :, :, 1] = ((grid[:, :, :, 1] + 1) * H - 1) / 2

        sampled_tensors = []
        for b in range(B):
            sampled_channels = []
            for c in range(C):
                channel = input_tensor[b, c]
                x_coords = grid[b, :, :, 0].flatten()
                y_coords = grid[b, :, :, 1].flatten()
                coords = np.stack([y_coords, x_coords], axis=-1)
                sampled_channel = map_coordinates(channel, coords.T, order=1, mode='constant', cval=0.0).reshape(H, W)
                sampled_channels.append(sampled_channel)
            sampled_tensors.append(np.stack(sampled_channels, axis=0))

        return np.stack(sampled_tensors, axis=0)


def my_map_coordinates(input, coordinates):
    def set_value(input, x, y):
        mask = (x >= 0) & (x < input.shape[1]) & (y >= 0) & (y < input.shape[0])
        out = np.zeros(y.shape)
        out[mask] = input[y[mask], x[mask]]
        return out

    y = coordinates[0,:]
    x = coordinates[1,:]
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1
    
    f_x0_y0 = set_value(input, x0, y0)
    f_x1_y0 = set_value(input, x1, y0)
    f_x0_y1 = set_value(input, x0, y1)
    f_x1_y1 = set_value(input, x1, y1)

    denom = (y1-y0)*(x1-x0)
    f = ((y1-y)*(x1-x) / denom) * f_x0_y0 + ((y1-y)*(x-x0) / denom) * f_x1_y0 + ((y-y0)*(x1-x) / denom) * f_x0_y1 + ((y-y0)*(x-x0) / denom) * f_x1_y1
    return f.astype(np.float32)

if __name__=='__main__':
    model = UVDocPredictor('weights/uvdoc.onnx')
    img = cv2.imread('images/demo3.jpg')
    out_img = model.predict(img)
    cv2.imwrite('unwrap_predictor_out.jpg', out_img)