import tenseal as ts
import torch

from client import client_query, context
from server import process_query


print("\nSending encrypted query to server...")


server_response = process_query(
    client_query
)


print("\nServer finished encrypted inference.")


print("\nDecrypting result...")


result = ts.ckks_vector_from(
    context,
    server_response["data"]
).decrypt()


probabilities = torch.softmax(
    torch.tensor(result),
    dim=0
)


prediction = torch.argmax(probabilities)


print("\nPrediction:")
print(prediction.item())


print("\nProbabilities:")
print(probabilities)