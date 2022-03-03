from itertools import count
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature


class SkyLeftAirplaneChallenger:
    """A fast solution for identifying vertical rivers"""
    def __init__(self):
        self.flag = "skyleftairplane_model"
        self.sky_threshold = 1800
        self.debug = False

    @staticmethod
    def _remove_border(img):
        img[:, 1] = 0
        img[:, -2] = 0
        img[1, :] = 0
        img[-2, :] = 0
        return img

    def solution(self, img_stream, **kwargs) -> bool:  # noqa
        """Implementation process of solution"""
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        edges1 = feature.canny(img)
        edges1 = self._remove_border(edges1)
        edges2 = feature.canny(img, sigma=3)
        edges2 = self._remove_border(edges2)

        # display results
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

        # ax[0].imshow(img, cmap='gray')
        # ax[0].set_title('noisy image', fontsize=20)

        # ax[1].imshow(edges1, cmap='gray')
        # ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

        # ax[2].imshow(edges2, cmap='gray')
        # ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

        # for a in ax:
        #     a.axis('off')

        # fig.tight_layout()
        # plt.show()

        # fill_plane = ndi.binary_fill_holes(edges1)

        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.imshow(fill_plane, cmap=plt.cm.gray)
        # ax.set_title('filling the holes')
        # ax.axis('off')
        # plt.show()
        # print(np.count_nonzero(edges1))
        # print(np.count_nonzero(edges2))

        if np.count_nonzero(edges1) > self.sky_threshold:
            if self.debug:
                print('[not in sky] ', end='')
            return False

        # get avg coordinate of edges where edges are not zero
        # avg_point = np.average(np.nonzero(edges1), axis=1)
        # print(avg_point)

        min_x = np.min(np.nonzero(edges1), axis=1)[1]
        max_x = np.max(np.nonzero(edges1), axis=1)[1]

        left_nonzero = np.count_nonzero(edges1[:, min_x:min(max_x, min_x + 10)])
        right_nonzero = np.count_nonzero(edges1[:, max(min_x, max_x - 10):max_x])

        # print(left_nonzero, right_nonzero)

        if left_nonzero > right_nonzero:
            if self.debug:
                print('[not turn left] ', end='')
            return False

        # mid_x = (min_x + max_x) / 2

        # print(min_x, max_x, mid_x, avg_point[0] < mid_x)

        # if avg_point[0] >= mid_x:
        #     return False

        # plt.show()
        return True


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result_path = 'result.txt'
    if os.path.exists(result_path):
        os.remove(result_path)

    # result_file = open(result_path, 'w')
    result_file = sys.stdout

    base_path = os.path.join('..', 'database', 'airplane_in_the_sky_flying_left')
    image_list = os.listdir(base_path)
    # image_list.sort()
    for image_name in image_list:
        image_path = os.path.join(base_path, image_name)
        with open(image_path, "rb") as file:
            data = file.read()
        solution = SkyLeftAirplaneChallenger().solution(data)
        result_file.write(f'{image_name}: {solution}\n')
        result_file.flush()

    result_file.close()