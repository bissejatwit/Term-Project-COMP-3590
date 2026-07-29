import tenseal as ts
import torch
from torchvision import transforms
from random import randint
from PIL import Image


def create_ctx():
    """
    Create and return a CKKS context.
    """

    poly_mod_degree = 8192

    coeff_mod_bit_sizes = [
        40, 21, 21, 21,
        21, 21, 21, 40
    ]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_mod_degree,
        -1,
        coeff_mod_bit_sizes
    )

    context.global_scale = 2 ** 21
    context.generate_galois_keys()

    return context


def load_input():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    idx = randint(1, 5)

    img_name = f"data/mnist-samples/img_{idx}.jpg"

    print("Loading:", img_name)

    img = Image.open(img_name)

    tensor_img = transform(img)

    return tensor_img.view(28, 28).tolist(), img

def prepare_input(ctx, plain_input):
    """
    Encode and encrypt a 28x28 image for CNN inference.
    """

    enc_input, windows_nb = ts.im2col_encoding(
        ctx,
        plain_input,
        7,
        7,
        3
    )

    assert windows_nb == 64

    return enc_input