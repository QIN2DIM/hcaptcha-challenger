import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from scipy.cluster.vq import kmeans2


class ElephantSolution:

    def __init__(self):
        self.debug = True

    def solution(self, img_stream, **kwargs) -> bool:  # noqa
        """Implementation process of solution"""
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("img", img)
        cv2.waitKey(0)

        if not self._style_classification(img):
            return False

        # using model to predict

        return True

    def _style_classification(self, img):
        # opencv to numpy
        img = np.array(img)
        # from 3d -> 2d
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2])).astype(np.float64)
        # print(img.shape)
        centroid, label = kmeans2(img, k=3)
        # print(centroid)
        # print(label)

        green_centroid = np.array([0.0, 255.0, 0.0])

        flag = False
        min_dis = np.inf
        for i in range(len(centroid)):
            # distance between centroid and green < threshold
            # print(np.linalg.norm(centroid[i] - green_centroid))
            min_dis = min(min_dis, np.linalg.norm(centroid[i] - green_centroid))

        if min_dis < 200:
            flag = True

        return flag


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result_path = 'result.txt'
    if os.path.exists(result_path):
        os.remove(result_path)

    # result_file = open(result_path, 'w')
    result_file = sys.stdout

    base_path = os.path.join('..', 'database', 'elephants_drawn_with_leaves')
    image_list = os.listdir(base_path)
    # image_list.sort()
    for image_name in image_list:
        image_path = os.path.join(base_path, image_name)
        with open(image_path, "rb") as file:
            data = file.read()
        solution = ElephantSolution().solution(data)
        result_file.write(f'{image_name}: {solution}\n')
        result_file.flush()

    result_file.close()