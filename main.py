
import pixellib
from pixellib.semantic import semantic_segmentation
import cv2
import numpy as np

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

img_path = "dataset/OurDataset/071/071.jpg"
segvalues, output, masks = segment_image.segmentAsAde20k(img_path, overlay=True)
cv2.imwrite("result/seg_result/sss.jpg", output)

print('=====================================')
# dict_keys(['class_ids', 'masks'])
print(segvalues.keys())
# <class 'list'>
print(type(masks))
# dict_keys(['class', 'name', 'color', 'mask', 'ratio'])
print(masks[0].keys())

semantic_num = len(segvalues['class_ids'])

for i in range(semantic_num):
  
  data = np.zeros((masks[0]['mask'].shape[:2]), dtype=np.uint8)
  name = masks[i]['name']
  mask = masks[i]['mask']
  data[mask == True] = 255
  cv2.imwrite(f'result/seg_result/{name}.jpg', data)