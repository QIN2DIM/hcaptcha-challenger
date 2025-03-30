from __future__ import annotations

import gzip
import html
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Union, Iterable

import cv2
import ftfy
import numpy as np
import regex as re
from PIL import Image
from onnxruntime import InferenceSession


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class Tokenizer:
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(
        self, texts: Union[str, Iterable[str]], context_length: int = 77, *args, **kwargs
    ) -> np.array:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            result[i, : len(tokens)] = np.array(tokens)

        return result


class Preprocessor:
    """
    Our approach to the CLIP `preprocess` neural net that does not rely on PyTorch.
    The two approaches fully match.
    """

    # Fixed variables that ensure the correct output shapes and values for the `Model` class.
    CLIP_INPUT_SIZE = 224
    # Normalization constants taken from original CLIP:
    # https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L85
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 3))
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 3))

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """Resize and crop an image to a square, preserving the aspect ratio."""

        # Current height and width
        h, w = img.shape[0:2]

        if h * w == 0:
            raise ValueError(
                f"Height and width of the image should both be non-zero but got shape {h, w}"
            )

        target_size = Preprocessor.CLIP_INPUT_SIZE

        # Resize so that the smaller dimension matches the required input size.
        # Matches PyTorch:
        # https://github.com/pytorch/vision/blob/7cf0f4cc1801ff1892007c7a11f7c35d8dfb7fd0/torchvision/transforms/functional.py#L366
        if h < w:
            resized_h = target_size
            resized_w = int(resized_h * w / h)
        else:
            resized_w = target_size
            resized_h = int(resized_w * h / w)

        # PIL resizing behaves slightly differently than OpenCV because of
        # antialiasing. See also
        # https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html
        # CLIP uses PIL, so we do too to match its results. But if you don't
        # want to have PIL as a dependency, feel free to change the code to
        # use the other branch.
        use_pil_for_resizing = True

        if use_pil_for_resizing:
            # https://github.com/pytorch/vision/blob/7cf0f4cc1801ff1892007c7a11f7c35d8dfb7fd0/torchvision/transforms/functional_pil.py#L240
            # We're working with float images but PIL uses uint8, so convert
            # there and back again afterwards
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((resized_w, resized_h), resample=Image.BICUBIC)
            img = np.array(img_pil).astype(np.float32) / 255
        else:
            img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

        # Now crop to a square
        y_from = (resized_h - target_size) // 2
        x_from = (resized_w - target_size) // 2
        img = img[y_from : y_from + target_size, x_from : x_from + target_size, :]

        return img

    @staticmethod
    def _image_to_float_array(img: Union[Image.Image, np.ndarray]):
        """Converts a PIL image or a NumPy array to standard form.

        Standard form means:
        - the shape is (H, W, 3)
        - the dtype is np.float32
        - all values are in [0, 1]
        - there are no NaN values

        Args:
            img: The image to convert.

        Returns:
            The image converted to a NumPy array in standard form.

        Raises:
            ValueError if the image is invalid (wrong shape, invalid
                values...).
        """
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(f"Expected PIL Image or np.ndarray but instead got {type(img)}")

        if isinstance(img, Image.Image):
            # Convert to NumPy
            img = np.array(img)

        if len(img.shape) > 3:
            raise ValueError(
                f"The image should have 2 or 3 dimensions but got {len(img.shape)} dimensions"
            )
        if len(img.shape) == 3 and img.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image but got image with {img.shape[2]} channels"
            )

        # Handle grayscale
        if len(img.shape) == 2:
            # The model doesn't support HxWx1 images as input
            img = np.expand_dims(img, axis=2)  # HxWx1
            img = np.concatenate((img,) * 3, axis=2)  # HxWx3

        # At this point, `img` has the shape (H, W, 3).

        if np.min(img) < 0:
            raise ValueError(
                "Images should have non-negative pixel values, "
                f"but the minimum value is {np.min(img)}"
            )

        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) > 1:
                raise ValueError(
                    "Images with a floating dtype should have values "
                    f"in [0, 1], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32)
        elif np.issubdtype(img.dtype, np.integer):
            if np.max(img) > 255:
                raise ValueError(
                    "Images with an integer dtype should have values "
                    f"in [0, 255], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32) / 255
            img = np.clip(img, 0, 1)  # In case of rounding errors
        else:
            raise ValueError(f"The image has an unsupported dtype: {img.dtype}.")

        if np.isnan(img).any():
            raise ValueError(f"The image contains NaN values.")

        try:
            # These should never trigger, but let's do a sanity check
            assert np.min(img) >= 0
            assert np.max(img) <= 1
            assert img.dtype == np.float32
            assert len(img.shape) == 3
            assert img.shape[2] == 3
        except AssertionError as e:
            raise RuntimeError(
                "Internal preprocessing error. The image does not have the expected format."
            ) from e

        return img

    def __call__(self, img: Union[Image.Image, np.ndarray], *args, **kwargs) -> np.ndarray:
        """Preprocesses the images like CLIP's preprocess() function:

        Args:
            img: PIL image or numpy array

        Returns:
            img: numpy image after resizing, center cropping and normalization.
        """
        img = Preprocessor._image_to_float_array(img)

        img = Preprocessor._crop_and_resize(img)

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img


@dataclass
class MossCLIP:
    visual_session: InferenceSession
    textual_session: InferenceSession

    _tokenizer = None
    _preprocessor = None

    def __post_init__(self):
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    @classmethod
    def from_pluggable_model(cls, visual_model: InferenceSession, textual_model: InferenceSession):
        return cls(visual_session=visual_model, textual_session=textual_model)

    def encode_image(self, images: Iterable[Image.Image | np.ndarray]) -> np.ndarray:
        """
        Compute the embeddings for a list of images.

        :param images:
            A list of images to run on. Each image must be a 3-channel(RGB) image.
            Can be any size, as the preprocessing step will resize each image to size (224, 224).
        :return:
            An array of embeddings of shape (len(images), embedding_size).

        """
        images = [self._preprocessor(image) for image in images]
        batch = np.concatenate(images)
        input_name = self.visual_session.get_inputs()[0].name
        return self.visual_session.run(None, {input_name: batch})[0]

    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        """
        Compute the embeddings for a list of texts.

        :param texts:
            A list of texts to run on. Each entry can be at most 77 characters.
        :return:
            An array of embeddings of shape (len(texts), embedding_size).
        """
        text = self._tokenizer(texts)
        input_name = self.textual_session.get_inputs()[0].name
        return self.textual_session.run(None, {input_name: text})[0]

    def __call__(
        self, images: Iterable[Image.Image | np.ndarray], candidate_labels, *args, **kwargs
    ):
        """

        :param images:
        :param candidate_labels:
        :param args:
        :param kwargs:
        :return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:

            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
        """

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        image_features = self.encode_image(images)
        text_features = self.encode_text(candidate_labels)

        image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)

        text_probs = 100 * image_features @ text_features.T
        text_probs = softmax(text_probs)

        result = [
            {"score": score, "label": label}
            for score, label in sorted(zip(text_probs[0], candidate_labels), key=lambda x: -x[0])
        ]
        return result
