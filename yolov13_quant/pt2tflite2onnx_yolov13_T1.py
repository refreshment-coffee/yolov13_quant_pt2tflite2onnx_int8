
#!/usr/bin/env Python
# coding=utf-8
from ultralytics import YOLO
import subprocess
import os
import sys


# calibration_file = "./calibration_image_sample_data_20x128x128x3_float32.npy"


#### export.py代码，注意改了弄回去
# model_path = r"/home/xxxxxxx/yolov13_test/yolov13/quant/1215_t3_quant/yolov13_dlchj_yolov13_1280_MAP_989_979_best.pt"
model_path=r"/home/xxxxxxx/yolov13_best.pt"
model_name=os.path.basename(model_path).split(".")[0]
root=os.path.dirname(model_path)

yaml=r"/home/xxxxxxx/datasets/data.yaml"
# # # Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)
# # # # Use the model
print("model_loaded!!!")

###!!!!!!!! 后续改的时候，如下路径也要进行修改
# /home/XXXXX/anaconda3/envs/XXXXX/lib/python3.10/site-packages/ultralytics/engine/exporter.py--L453
#     def get_int8_calibration_dataloader(self, prefix=""):
#         """Build and return a dataloader suitable for calibration of INT8 models."""

#         self.args.data=r"/home/XXXXXX/datasets/XXXXX.yaml"



try: 
    success = model.export(format="tflite",opset=11,dynamic=False,int8=True)  # export the model to ONNX format   # 指定本地校准样本
except Exception as e:
    print("Error : e")
print("--------------------------------------------------")

other_file=os.path.join(root,model_name+"_saved_model")
tflite_path = os.path.join(other_file,model_name+"_integer_quant.tflite")
quant_onnx_path =  os.path.join(root,model_name+"_INT8.onnx")

print(tflite_path)
if os.path.exists(tflite_path):
    print("\n\n\ntf2onnx ....\n\n\n")
    command = r"python -m tf2onnx.convert --opset 13 --tflite {}  --output {}  --outputs PartitionedCall:0,PartitionedCall:1,PartitionedCall:2  --outputs-as-nchw PartitionedCall:0,PartitionedCall:1,PartitionedCall:2   --inputs serving_default_images:0   --inputs-as-nchw serving_default_images:0".format(tflite_path,quant_onnx_path)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in process.stdout:
        print(line.rstrip())

    for line in process.stderr:
        print(line.rstrip(), file=sys.stderr)
        
    process.wait()

