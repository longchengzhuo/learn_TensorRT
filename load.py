import cv2
import torch
import numpy as np
from collections import OrderedDict
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
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


# def create_trt(trt_file, input_tensor_name, max_batch_size, device):
#     trt_data_type_to_torch = {
#         trt.float32: torch.float32,
#         trt.float16: torch.float16,
#         trt.int32: torch.int32,
#         trt.int8: torch.int8,
#     }
#     logger = trt.Logger(
#         trt.Logger.ERROR)  # create Logger, available level: VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR
#     with open(trt_file, "rb") as f:
#         engine_bytes = f.read()
#     engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)  # create inference engine
#     context = engine.create_execution_context()  # create Execution Context from the engine (analogy to a GPU context, or a CPU process)
#     tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
#     context.set_input_shape(input_tensor_name, (
#     max_batch_size, 9, 288, 512))  # set runtime size of input tensor if using Dynamic-Shape mode
#     buffer = OrderedDict()  # prepare the memory buffer on host and device
#     for name in tensor_name_list:  # 2 names: input; output
#         data_type = engine.get_tensor_dtype(name)
#         runtime_shape = context.get_tensor_shape(name)
#         buffer[name] = torch.empty(tuple(runtime_shape), dtype=torch.float32, device=device)
#     for name in tensor_name_list:
#         pass
#         # context.set_tensor_address(name, buffer[name].data_ptr())
#     return context