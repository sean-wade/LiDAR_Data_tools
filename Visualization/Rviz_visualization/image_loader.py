#!/usr/bin/python2
# -*- coding:utf-8 -*-

import cv2
import numpy as np


class ImageDataLoader(object):
    def __init__(self, image_files):
        """[summary]

        Args:
            image_files (list): [list of full_path with .jpg/.png extension files]
            # ! < MUST SORT > !
        """
        self.idx = 0
        self.image_files = image_files
        self.image_nums = len(self.image_files)
        print("ImageDataLoader Init with [%d] files"%self.image_nums)

    
    def __len__(self):
        return self.image_nums


    def __iter__(self):
        return self

    
    def __next__(self):
        """[summary]
        Returns:
            cv2.image
        """
        if self.idx == self.image_nums:
            raise StopIteration

        image_path = self.image_files[self.idx]
        image = self.get_data(image_path)

        self.idx += 1
        return image


    def load_idx(self, idx):
        if idx >= self.image_nums:
            print("Load image index[%d] out of range[%d] !!!"%(idx, self.image_nums))
            return None

        image_path = self.image_files[idx]
        image = self.get_data(image_path)
        return image

    
    def get_data(self, image_path):
        print(image_path)
        img = cv2.imread(image_path)
        return np.empty(0) if img is None else img


if __name__ == "__main__":
    import os
    import glob

    # ! only test load, unsorted files.
    files = sorted(glob.glob("/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/cam60/*"), key=lambda x: float(x[55:72]))
    image_generator = ImageDataLoader(files)


    print("Total image files = ", len(image_generator))
    # for image in image_generator:
    #     print(image.shape)

    iii = 0
    while True:
        # image = next(image_generator)
        # image = image_generator.__next__()
        image = image_generator.load_idx(iii)
        print(image.shape)
        cv2.imshow(".", image)
        cv2.waitKey(100)
        iii += 1



