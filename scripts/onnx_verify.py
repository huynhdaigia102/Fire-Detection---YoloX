import onnxruntime as rt
sess = rt.InferenceSession("/mnt/sda2/ExternalHardrive/research/object_detection/common/yolox_original/weights/person/nano_20220627.onnx")
print("====INPUT====")
for i in sess.get_inputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====OUTPUT====")
for i in sess.get_outputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))