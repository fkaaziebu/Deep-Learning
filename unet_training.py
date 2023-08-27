from data.generate_unet_data import Generate
from models.unet import Unet
import matplotlib.pyplot as plt

DIR = "./dataset"

data = Generate(src=DIR)
img_train, mask_train, img_test, mask_test, img_array, mask_array = data.forward()

model = Unet(
    input_size=(256, 256, 1),
    img_train=img_train,
    mask_train=mask_train,
    img_test=img_test,
    mask_test=mask_test,
    batch_size=8,
    num_epochs=15
)

# Training
history = model.forward()
#
# # Test on an image
# img_side_size = 256
# pred = model.segment_xray_image(img_array=img_array, img_num=1, img_side_size=img_side_size)
#
# fig = plt.figure(figsize=(15, 10))
# plt.imshow(pred.reshape(img_side_size, img_side_size))
# plt.title("Image segmentation")
# plt.axis("off")
# plt.show()
