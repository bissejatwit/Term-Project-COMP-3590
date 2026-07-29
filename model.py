import tenseal as ts
from typing import Dict


class ConvMNIST:
    """
    CNN used for encrypted MNIST inference.
    """

    def __init__(self, parameters: Dict[str, list]):
        """
        Store the pretrained model parameters.
        """

        self.conv1_weight = parameters["conv1_weight"]
        self.conv1_bias = parameters["conv1_bias"]

        self.fc1_weight = parameters["fc1_weight"]
        self.fc1_bias = parameters["fc1_bias"]

        self.fc2_weight = parameters["fc2_weight"]
        self.fc2_bias = parameters["fc2_bias"]

        self.windows_nb = parameters["windows_nb"]

    def forward(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        Perform encrypted inference.
        """

        channels = []

        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(
                kernel,
                self.windows_nb
            ) + bias

            channels.append(y)

        out = ts.CKKSVector.pack_vectors(channels)

        out.square_()

        out = out.mm_(self.fc1_weight) + self.fc1_bias

        out.square_()

        out = out.mm_(self.fc2_weight) + self.fc2_bias

        return out

    @staticmethod
    def prepare_input(context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        """
        Reconstruct the encrypted input sent by the client.
        """

        ctx = ts.context_from(context)

        enc_x = ts.ckks_vector_from(
            ctx,
            ckks_vector
        )

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)