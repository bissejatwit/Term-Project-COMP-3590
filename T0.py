import tenseal as ts

# The TenSEALContext is a special object that holds different encryption keys and parameters
# Creating a single TenSEALContext before doing encrypted computation
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
context

# TenSEALContext is now holding the secret key
public_context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
print("Is the context private?", ("Yes" if public_context.is_private() else "No"))
print("Is the context public?", ("Yes" if public_context.is_public() else "No"))

sk = public_context.secret_key()

# the context will drop the secret-key at this point
public_context.make_context_public()
print("Secret-key dropped")
print("Is the context private?", ("Yes" if public_context.is_private() else "No"))
print("Is the context public?", ("Yes" if public_context.is_public() else "No"))

# create an encrypted vector of integers
plain_vector = [60, 66, 73, 81, 90]
encrypted_vector = ts.bfv_vector(context, plain_vector)
print("We just encrypted our plaintext vector of size:", encrypted_vector.size())
encrypted_vector

#  we can do both addition, subtraction and multiplication with plain vectors.
add_result = encrypted_vector + [1, 2, 3, 4, 5]
print(add_result.decrypt())

sub_result = encrypted_vector - [1, 2, 3, 4, 5]
print(sub_result.decrypt())

mul_result = encrypted_vector * [1, 2, 3, 4, 5]
print(mul_result.decrypt())

# or with we can do both addition, subtraction and multiplication with other encrypted vectors
encrypted_add = add_result + sub_result
print(encrypted_add.decrypt())


encrypted_sub = encrypted_add - encrypted_vector
print(encrypted_sub.decrypt())


encrypted_mul = encrypted_add * encrypted_sub
print(encrypted_mul.decrypt())

# never encrypt plaintext values to evaluate them with ciphertexts if they don't need to be kept private
# c2p evaluations are more efficient than c2c
from time import time

t_start = time()
_ = encrypted_add * encrypted_mul
t_end = time()
print("c2c multiply time: {} ms".format((t_end - t_start) * 1000))

t_start = time()
_ = encrypted_add * [1, 2, 3, 4, 5]
t_end = time()
print("c2p multiply time: {} ms".format((t_end - t_start) * 1000))