import numpy as np
import time
import os
import cv2
import argparse
import torch
from utils import *
import pycuda.driver as cuda
import tensorrt as trt
cuda.init()

HEIGHT = 288
WIDTH = 512


class TensorRTInference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(list(size), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        self.ctx.push()

        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data)
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[-1].host, self.outputs[-1].device, self.stream)
        self.ctx.pop()
        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[-1].host

def start_trt_infer(args):
    # 获取参数
    num_frame = args.num_frame
    max_batch_size = args.max_batch_size
    input_tensor_name = args.input_tensor_name
    device = torch.device(args.gpu)
    trt_file = args.trt_file
    video_file = args.video_file
    batch_size = args.batch_size
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    threshold = args.threshold
    start_width = args.start_width
    end_width = args.end_width
    video_name = video_file.split('/')[-1][:-4]
    video_format = video_file.split('/')[-1][-3:]
    out_video_file = f'{save_dir}/{video_name}_pred{threshold}yutrt.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_ballyutrt.csv'

 
    # 创建trt
    trt_file = args.trt_file  # 装载模型权重
    trt_inference = TensorRTInference(trt_file)
    # 获取视频配置
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 准备csv写入
    f = open(out_csv_file, 'w')
    f.write('Frame,Visibility,X,Y\n')


    # 参数初始化
    one_batch_frame_num = num_frame * batch_size
    frame_count = 0
    it_is_last_batch = 0
    ratio_h = h / HEIGHT
    ratio_w = w / WIDTH
    start_width_in_output_tensor = int(start_width / ratio_w)  # 开始的宽度索引
    end_width_in_output_tensor = int(end_width / ratio_w)  # 结束的宽度索引
    success = True

    start_time = time.time()
    # 开始每一轮batch的推理
    while success:
        # 将一个batch所有图像保存在frame_queue中
        frame_queue = []
        frame_count, frame_queue, it_is_last_batch, inferred_frame_num = get_one_batch_frame(one_batch_frame_num, cap,
                                                                                             frame_count, frame_queue,
                                                                                             it_is_last_batch)
        if not frame_queue:
            break
        x = assemble_frames_by_batch(frame_queue, num_frame)
        output_data_cpu = trt_inference.infer(x)
        output_tensor = output_data_cpu.reshape(-1, HEIGHT, WIDTH)            # 推理

        for i in range(output_tensor.shape[0]):
            if it_is_last_batch and i < inferred_frame_num:
                # 当最后一轮的frame数小于one_batch_frame_num时，不对已经后处理过的帧进行后处理
                continue
            else:
                vis, cx_pred, cy_pred, img = post_processing(i, frame_count, output_tensor, start_width_in_output_tensor,
                                                                end_width_in_output_tensor, ratio_h, ratio_w, frame_queue,
                                                                threshold)
                f.write(f'{frame_count-one_batch_frame_num+i},{vis},{cx_pred},{cy_pred}\n')
                out.write(img)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds")
    # out.release()
    trt_inference.ctx.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, default="/ssd2/cz/TrackNetV3/bt_for_test/cam1_2024-07-31_10-35-42.mp4")
    parser.add_argument('--trt_file', type=str, default='/ssd2/cz/TrackNetV3/predict_mp4/model_bestfp16.trt')
    parser.add_argument('--save_dir', type=str, default='/ssd2/cz/TrackNetV3/predict_mp4')
    parser.add_argument('--gpu', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--start_width', type=int, default=0)
    parser.add_argument('--end_width', type=int, default=1720)
    parser.add_argument('--num_frame', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=1)                       # 该参数将决定buffer的大小; 通常和batch_size相等即可
    parser.add_argument('--input_tensor_name', type=str, default="input")              # 该字符串必须严格为创建onnx时使用的名称
    args = parser.parse_args()
    start_trt_infer(args)
    print('Done.')









