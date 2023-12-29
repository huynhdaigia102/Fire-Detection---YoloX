import onnxruntime as rt
import numpy as np
import torch
import timeit

from bytetrack._C import Engine

input_size = [288, 480]

model = Engine.load("/research/object_tracking/git/ByteTrack/weights/bytetrack_nano.plan")
batch_size = 1

weight_paths = "/research/object_tracking/git/ByteTrack/weights/bytetrack_nano.onnx"
ort_session = rt.InferenceSession(weight_paths)
input_name = ort_session.get_inputs()[0].name
print(input_name)

image_tensor = np.random.randn(2, 3, *input_size).astype(np.float32)
torch_tensor = torch.from_numpy(image_tensor.copy()).to("cuda")

onnx_output = ort_session.run(None, {input_name: image_tensor})[0]

for _ in range(100):
    t0 = timeit.default_timer()
    trt_output = model(torch_tensor)
    trt_output = trt_output[0].cpu().numpy()
    t1 = timeit.default_timer()
    print(t1-t0)

print(np.sum(np.abs(onnx_output - trt_output)))
print(onnx_output.shape, trt_output.shape)
