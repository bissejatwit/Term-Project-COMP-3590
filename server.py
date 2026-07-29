import pickle
import os

from model import ConvMNIST

def load_parameters(file_path: str) -> dict:
    """
    Load pretrained CNN parameters from a pickle file.
    """

    try:
        with open(file_path, "rb") as file:
            parameters = pickle.load(file)

        print(f"Model loaded from '{file_path}'")

    except OSError as error:
        print("Error loading model:", error)
        raise

    return parameters
PARAMETERS_PATH = "parameters/ConvMNIST-0.1.pickle"

parameters = load_parameters(PARAMETERS_PATH)

model = ConvMNIST(parameters)

def process_query(client_query: dict) -> dict:
    """
    Process an encrypted query from the client.
    """

    encrypted_query = model.prepare_input(
        client_query["context"],
        client_query["data"]
    )

    encrypted_result = model(encrypted_query)

    return {
        "data": encrypted_result.serialize()
    }