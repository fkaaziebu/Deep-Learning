import os, re
from glob import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import numpy as np


class Generate:
    def __init__(self, src):
        self.source = src
        self.related_paths = defaultdict(list)
        self.lung_image_paths = glob(os.path.join(self.source, "Lung Segmentation/CXR_png/*.png"))
        self.mask_image_paths = glob(os.path.join(self.source, "Lung Segmentation/masks/*.png"))

    def image_cleanup(self):
        for img_path in self.lung_image_paths:
            img_match = re.search("CXR_png/(.*)\.png$", img_path)
            if img_match:
                img_name = img_match.group(1)
            for mask_path in self.mask_image_paths:
                mask_match = re.search(img_name, mask_path)
                if mask_match:
                    self.related_paths["image_path"].append(img_path)
                    self.related_paths["mask_path"].append(mask_path)

        return pd.DataFrame.from_dict(self.related_paths)

    def prepare_train_test(self, resize_shape=tuple(), color_mode="gray"):
        img_array = list()
        mask_array = list()
        paths_df = self.image_cleanup()

        for image_path in tqdm(paths_df.image_path):
            resized_image = cv2.resize(cv2.imread(image_path), resize_shape)
            resized_image = resized_image / 255.
            if color_mode == "gray":
                img_array.append(resized_image[:, :, 0])
            elif color_mode == "rgb":
                img_array.append(resized_image[:, :, :])

        for mask_path in tqdm(paths_df.mask_path):
            resized_mask = cv2.resize(cv2.imread(mask_path), resize_shape)
            resized_mask = resized_mask / 255.
            mask_array.append(resized_mask[:, :, 0])

        return img_array, mask_array

    def forward(self):
        img_array, mask_array = self.prepare_train_test(resize_shape=(256, 256))

        img_train, img_test, mask_train, mask_test = train_test_split(
            img_array,
            mask_array,
            test_size=0.2,
            random_state=42
        )

        img_side_size = 256
        img_train = np.array(img_train).reshape(len(img_train), img_side_size, img_side_size)
        img_test = np.array(img_test).reshape(len(img_test), img_side_size, img_side_size)
        mask_train = np.array(mask_train).reshape(len(mask_train), img_side_size, img_side_size)
        mask_test = np.array(mask_test).reshape(len(mask_test), img_side_size, img_side_size)

        return img_train, mask_train, img_test, mask_test, img_array, mask_array
