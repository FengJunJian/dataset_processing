from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import mmcv
import os

absP='D:/mmdetection'#模型路径
config_file = os.path.join(absP,'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../../../data_processing_cache/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cpu'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
img=os.path.join('../012.jpg')#'absP,demo/demo.jpg'
result=inference_detector(model, img)
outImg=model.show_result(
img,
result)
#outImg1=show_result_pyplot(model,img,result)
#mmcv.imshow(outImg,'a')
mmcv.imwrite(outImg,'aa.jpg')
print(result)
