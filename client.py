import numpy as np
import tenseal as ts
import torch

from helpers import create_ctx
from helpers import load_input
from helpers import prepare_input

context = create_ctx()

image, original = load_input()

encrypted_image = prepare_input(
    context,
    image
)
print("Encrypted image created!")


server_context = context.copy()

server_context.make_context_public()

serialized_context = server_context.serialize()

serialized_image = encrypted_image.serialize()

client_query = {
    "context": serialized_context,
    "data": serialized_image
}
print("Client query created!")
