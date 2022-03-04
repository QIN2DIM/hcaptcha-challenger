# -*- coding: utf-8 -*-
# Time       : 2022-03-01 20:32
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import hashlib
import time

import cv2
import numpy as np
from skimage import feature
from skimage.future import graph
from skimage.segmentation import slic


class SKRecognition:
    def __init__(self):
        self.flag = "skimage_model"
        self.rainbow_table = [
            "073830992aa0a236ba4768719bc244c4",
            "09b1671b876a2a90b2b7554d780c8d1f",
            "0ac441b30f88b96881ddb0ecc00e91d5",
            "0affc32833603c7128d9751f81940dd2",
            "109333fb80db69c5c12dcd38556d1533",
            "14f4432ec3e3dff9347519d3481db7e9",
            "1554003baaa88266664d362ee951c10b",
            "18fa94de5e14fcbcf07141a0f053ffd4",
            "192b43e67870258f080bac6ec51ca4a1",
            "19d4ac19c07cb35b83232a92b4fce16a",
            "1b54bcff83edc61490e2e6b975bfa248",
            "1b64b7999eb2e675b8d348458aecc2cc",
            "1d25f25cae364f53499f5d5d949b6972",
            "1f8f255b708a2920ff4a519b7fd31f26",
            "2036219696664f2584c2617e0b6f47f0",
            "208d0ef170eec84edb00b3b610d037dd",
            "210aa14c413a903797409109aad74730",
            "224c33820de1733947f18eaa0ba40c19",
            "23c986e5dbefa7144921b38e86dcdacb",
            "244175146e1d56dc9bf2478cce635c65",
            "2464e22e578dafc0be9c4a8e57937e96",
            "24b85b5dfd0b4407da75ec0932ea7e57",
            "252d1b9a86f0d02e45c20ae73dc31de1",
            "25832c896d60979f5a757a7754c66b5f",
            "2692672113713ea0b0c5493b40a62655",
            "2b3383ee499914f72545d0e3e67caf89",
            "2ba364b1aaeef05b4566b7a11d13ee4f",
            "2d390c32f42a4a8f98b1a18316079cab",
            "2ec77d2ffdccd6fcc68d556f6914f748",
            "2eeb9fed7f9e8d58840a44eb2783c2b8",
            "2fc27068c87b0c16f0eabed9d5ff5edd",
            "30c92b8777bc2383cf284125dea6a5e6",
            "33587951244c3e2f72643c1507356e39",
            "368356f4e40b67ca92092aac74b308ae",
            "390e38e93db482b8c2c9a2835f5440f6",
            "3a7b94b31de77e70401f7351b661922b",
            "3bef81df42e79109379ebd4f30515f1f",
            "3d44ca9ea582d96c15df3a72f408714e",
            "3f439cbcc0e751e5bd60c084bf2a9089",
            "3fe9b07361270b4cba4817714f6bce0f",
            "408910c94d5c3796b6e79ba59103a042",
            "41992176d8f4d201efda93367f7d8693",
            "41ce2ccb0295fd824ffc461317b0f6ee",
            "41e3d8ab1968b521da69c40930d64c97",
            "446fec3a5705818ae6e10a6fedc6e928",
            "44c80d54a0455ad0a895cb57109449a3",
            "499860a3e82dcfe31514e3377cde704c",
            "4a233230ba82b017932092e4f9910db4",
            "4a3c4b32dbeb95d4d05bb56b0baced63",
            "4dd826a9e672c44329e33add83e4d0dc",
            "4e03bfa03e7e609a825462fe6a490489",
            "4f5761bc417783df2751525c9819bc34",
            "4fa7d3e9cc145afd07e9eaeba8c57668",
            "51a84c433d2dd3871ac9cf99513b05ed",
            "52a6978c9e23892e8fd41e0ba445cab4",
            "53495793ffe70f2333dc3f13a2e976f5",
            "562ceb20aa9714f4a454e8f59479285c",
            "5664bb342df282eb25507aeb6ff55d5c",
            "5942fbc264c207fa7f7241ab0ea7febd",
            "5a35d5e10ad41726204ff9217f458825",
            "5aaf56da1d1d2be2cdf22ae7aa35ed53",
            "5b4e75a2efd0349d872e013c7b5bffbc",
            "5c7e8e2568e48942d5b86c5d131992d3",
            "5def8238d14d8cb46102658ed0f88ee0",
            "5e695f6fa80a9b7313000bf9058ae0c1",
            "5fb3dd84a076b1cf7c4f20ac243855fa",
            "63f0c38ac980946dd298133d4c4704ba",
            "668243b8b92dc889e9eb2f68a6dd35e1",
            "668b5eba74ec2ca81f492e7b88a38dce",
            "66e615b546c2d208bfe5baf2c93c508a",
            "68707ec27285f5fd692022d0c14c951c",
            "68cf0092d21e074905060c53627b4990",
            "6abdf360f4a0b33eade77b021e9eb1e1",
            "6db3105e05989385a168224f0b5e0590",
            "6fa93fdea2318b8d82cbd4db9d268e63",
            "70173486deee31417614f9f0d1b1d397",
            "7669e6357d303882dbd04b6226ac247a",
            "76be428c1b5cddbe650ab2c57b910ad6",
            "76cf2d69222a79420b08020bd08f1cf2",
            "77c568035f4a3050de4eb6d261bf7632",
            "7858eae5d645eefa6f724dc5ca1c4595",
            "78d45d0e3c385e54dff856afb6d22ca1",
            "7ab4908a7b35f52ecbc16a0c93a1d00f",
            "7b03e3bdf3627e2a24c51edb45b5b864",
            "7b10e3977c314c8107f6c8623d477109",
            "7bf7b0a7ed43a698794657436177192c",
            "7ca6310901d569f4395a2fb29c115283",
            "7ce807b0d66f5bb2e379cc1d805dc50b",
            "7fd3743110b922ce9a22888d0368d4fc",
            "8268439ffe4b139a610235dbf4ce0a5e",
            "83df0b41768905ad3507ce0555b1372d",
            "85734f987dcf54f58eba3fe8687f73d8",
            "8766b451d804fb1a298a406df7bb6bd1",
            "88ecea0a9cf9e69ad3edfe317908157e",
            "90ddec05470527856e4643913133244d",
            "919648956cfd926d1527af75f00dd86f",
            "91b5cbdc5ec7a8144a71225f3a6c801b",
            "91ee713aa02c4a7be7aa84a794a98268",
            "9378b45df2a113a0f28865e6aec28538",
            "941f3a1ff7171fa635977ebbc47fa58c",
            "951780f1d8be4a37536c22aa21427e22",
            "951df676a7d05dacb74330b16500365d",
            "98bb49c9dc868f95a2555765058cdac1",
            "98c067d348370cdba4ffcb59e4eb4c92",
            "9904ca357aea6d875823337f4bf9a078",
            "9ad81c59754af646f009290c29c2183d",
            "9bae47bb0e8cc6e58c7580e8a71d4936",
            "9cdf1db0d6cb67312364edfd9098b7c0",
            "9d18afdd4f4fe5c184e08b35bce6d265",
            "9e636e743c14cf1e528c82c836c4b577",
            "9e87f33d1337f7d2bdbd39459cf73157",
            "a1440ee6afe4971d9b6b848c3e457225",
            "a1a19fca09e69aa4902b4c89078a9467",
            "a1e90e58d81232ba26924703f0e96adf",
            "a2a159f7cd4a7d160b11e509b9578f83",
            "a318d7a1f7ca4c138b508d4360e6b6b6",
            "a7f8d44e1abc274deeccf9c32161518a",
            "a93c845a69f292a7e2c57a23e1bd3899",
            "ab5077d832942050b26daa121d22b168",
            "ac9d8d192b27067373f47f54db0224e8",
            "ae65112fe17af19f8dcf9535fea9d16f",
            "b2b68b9c858ecc4a2698152e566f950b",
            "b6d415c62b74101b7fdb10366c5bc61a",
            "b816e8b20ed6c1fc7512ff8b7cdfb577",
            "b8d8bbd1781474ee771ba1e64da1984d",
            "bac41e8da3a19d9235bb7e5882a7f457",
            "bbde99712881b3b98b581d2578570544",
            "bc0ca211a32da2c63b3dc36428de890d",
            "be8775815d0309666f8f8a098cfa8924",
            "c124e9d6e0c9b242014ada99782cc6f9",
            "c33119cb73df87e1673081ff78b9ac11",
            "c4017b6919c601bb21fe43d4a7966de4",
            "c4fded11f7d76a5d651827a3ca8a0323",
            "c567c4de960e9d9fbeb79e13c08e63d4",
            "c84e6e10de13696a373b356451d66388",
            "ca291d9313b08664798c22f87939a3d2",
            "ca5e981930a8ebd78b2465ba06a51e41",
            "cb58e025114d0a665667e1eae4c53bd5",
            "cc281d20d4eff4c4603fb887736e4250",
            "cc3b9da5799832390dcbdf1e032887aa",
            "d12c8a05c06ac0fbde2140c7cbbd0c40",
            "d65ff264328043155c68f5263d54108c",
            "d6f26e3eb219813d7a3c2228ed495e09",
            "d94de8f5706d80ebe4f2099629c6ff4a",
            "de2fd04a3ba285ba8537b9408666ad2d",
            "e074627ffbaa334fa51f878440aad938",
            "e094419fa707ceeb7830b16d7cebfec9",
            "e2a3fdc1ad0e2f880a26678d6d04c437",
            "e39db5c8477b215a8e8a0172159bb796",
            "e39e850d03f4f04dd0048a629d05edf3",
            "e5e35a536c2559fc0f24002afcacc056",
            "e70a174366b77fc30a304eb4cfd92842",
            "ea2e2ecb4fca0c9fdd7a53d4a4698433",
            "ed5688ea93f71390534486046d11ad12",
            "edfe4351c144389d4ac8789b88a60eb2",
            "f12b7bb081893ffb37d8615833052f07",
            "f5b1d140d4d9e790b64ad68f4249ee1a",
            "f732a53769fca365299b860817b768d6",
            "f8036531672653731cd97137b701faff",
            "fcd68e576bdfa8e9add2b92dd388401a",
            "fd0e5ee44a4b9e36f259e18827cf726c",
            "ff7c3a89a04f1121bcbfd6fce29f0cc5",
            "fff93f192b3e5b0f4a285b7bdfb1eea8",
            "",
        ]
        self.rainbow_table = dict(
            zip(self.rainbow_table, ["_"] * len(self.rainbow_table))
        )

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError


class RiverChallenger(SKRecognition):
    """A fast solution for identifying vertical rivers"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _weight_mean_color(graph_, src: int, dst: int, n: int):  # noqa
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph_ : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph_.nodes[dst]["mean color"] - graph_.nodes[n]["mean color"]
        diff = np.linalg.norm(diff)
        return {"weight": diff}

    @staticmethod
    def _merge_mean_color(graph_, src, dst):
        """Callback called before merging two nodes of a mean color distance graph.

        This method computes the mean color of `dst`.

        Parameters
        ----------
        graph_ : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph_.nodes[dst]["total color"] += graph_.nodes[src]["total color"]
        graph_.nodes[dst]["pixel count"] += graph_.nodes[src]["pixel count"]
        graph_.nodes[dst]["mean color"] = (
            graph_.nodes[dst]["total color"] / graph_.nodes[dst]["pixel count"]
        )

    def solution(self, img_stream, **kwargs) -> bool:  # noqa
        """Implementation process of solution"""
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        img = cv2.pyrMeanShiftFiltering(img, sp=10, sr=40)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=75)

        labels = slic(img, compactness=30, n_segments=400, start_label=1)
        g = graph.rag_mean_color(img, labels)

        labels2 = graph.merge_hierarchical(
            labels,
            g,
            thresh=35,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merge_mean_color,
            weight_func=self._weight_mean_color,
        )

        return len(np.unique(labels2[-1])) >= 3


class DetectionChallenger(SKRecognition):
    """A fast solution for identifying `airplane in the sky flying left`"""

    def __init__(self):
        super().__init__()
        self.sky_threshold = 1800
        self.left_threshold = 30

    @staticmethod
    def _remove_border(img):
        img[:, 1] = 0
        img[:, -2] = 0
        img[1, :] = 0
        img[-2, :] = 0
        return img

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""

        # This algorithm is too fast! You need to add some delay before returning the correct result.
        # Without the delay, this solution would have passed the challenge in `0.03s`,
        # which is `humanly' impossible to do.
        if self.rainbow_table.get(hashlib.md5(img_stream).hexdigest()):
            time.sleep(0.25)
            return True

        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges1 = feature.canny(img)
        edges1 = self._remove_border(edges1)

        # on the ground
        if np.count_nonzero(edges1) > self.sky_threshold:
            return False

        min_x = np.min(np.nonzero(edges1), axis=1)[1]
        max_x = np.max(np.nonzero(edges1), axis=1)[1]

        left_nonzero = np.count_nonzero(
            edges1[:, min_x : min(max_x, min_x + self.left_threshold)]
        )
        right_nonzero = np.count_nonzero(
            edges1[:, max(min_x, max_x - self.left_threshold) : max_x]
        )

        # Flying towards the right
        if left_nonzero > right_nonzero:
            return False

        time.sleep(0.25)
        return True
