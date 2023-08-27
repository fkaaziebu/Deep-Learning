from models.unet import Unet
from data.generate_unet_data import Generate
import matplotlib.pyplot as plt

DIR = "./dataset"

data1 = Generate(src=DIR)
img_train, mask_train, img_test, mask_test, img_array, mask_array = data1.forward()

# unet_seg = Unet(
#     input_size=(256, 256, 1),
#     img_train="img_train",
#     mask_train="mask_train",
#     img_test="img_test",
#     mask_test="mask_test",
#     batch_size=16,
#     num_epochs=1
# )
#
# unet_seg.load_weights()

# Print out a single image array
# print(img_array[0])

# Segmentation
# img_side_size = 256
# input_img = "./dataset/CXR_png/"
# input_img = unet_seg.predict(input_img=input_img)
# input_img = unet_seg.segment_xray_image(img_array=img_array, img_num=0)
fig = plt.figure(figsize=(15, 10))
plt.imshow(img_array[3])
plt.title("Image to segment")
plt.axis("off")
plt.show()