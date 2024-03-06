Forked from ppogg - here is the repo: https://github.com/ppogg/YOLOv5-Lite

# Changes
CCTV - Delayed real-time object detection the more time we run the model the more it delays.
Fix: https://github.com/ultralytics/yolov5/issues/4465#issuecomment-1113038325

datasets.py:362 comment  time.sleep(1 / self.fps[i]) # wait time

error: AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
Fix:  https://blog.csdn.net/weixin_43401024/article/details/124428432

C:\Users\jhojh\Anaconda3\envs\yolov5lite\lib\site-packages\torch\nn\modules\module.py:153-154
replace `
def forward(self, input: Tensor) -> Tensor:
        # return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
        #                      recompute_scale_factor=self.recompute_scale_factor)
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
`

Fixed Distance Estimation
Detect.py: 
added width per class
#line 310 added read focal method



Test
python test.py --weights weights/vih-model.pt --data vih-dataset/data.yaml --verbose --name addPerClass
python test.py --weights weights/wcm-model.pt --data wcm-dataset/data.yaml --verbose --name addPerClass
