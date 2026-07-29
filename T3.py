import tenseal as ts
import tenseal.sealapi as seal
import random
import pickle
import numpy as np
import math
from IPython.display import HTML, display
import tabulate
import pytest

# setting up the environment and defining the encryption settings

#convert bytes into human readable format 
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

enc_type_str = {
    ts.ENCRYPTION_TYPE.SYMMETRIC : "symmetric", 
    ts.ENCRYPTION_TYPE.ASYMMETRIC : "asymmetric",
}

scheme_str = {
    ts.SCHEME_TYPE.CKKS : "ckks",  # encryption scheme designed for decimal (floating point) numbers.
    ts.SCHEME_TYPE.BFV : "bfv",
}

def decrypt(enc):
    return enc.decrypt()

# How much storage does an encryption context take with different settings
# creates different encryption contexts, measures their size, and stores the results in a table

ctx_size_benchmarks = [["Encryption Type", "Scheme Type", "Polynomial modulus", "Coefficient modulus sizes", "Saved keys", "Context serialized size", ]]

# testing both symmetric and asymmetric encryption on everything 
for enc_type in [ts.ENCRYPTION_TYPE.SYMMETRIC, ts.ENCRYPTION_TYPE.ASYMMETRIC]:
    for (poly_mod, coeff_mod_bit_sizes) in [
        (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
        (8192, [40, 20, 40]),
        (8192, [20, 20, 20]),
        (8192, [17, 17]),
        (4096, [40, 20, 40]),
        (4096, [30, 20, 30]),
        (4096, [20, 20, 20]),
        (4096, [19, 19, 19]),
        (4096, [18, 18, 18]),
        (4096, [18, 18]),
        (4096, [17, 17]),
        (2048, [20, 20]),
        (2048, [18, 18]),
        (2048, [16, 16]),
    ]:
        # Create a CKKS encryption context using these exact parameters
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        context.generate_galois_keys() # used for: rotating encrypted vectors, shifting encrypted values
        context.generate_relin_keys() # used for: needed after multiplying ciphertexts
        
        ser = context.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "all", convert_size(len(ser))])
        
        if enc_type is ts.ENCRYPTION_TYPE.ASYMMETRIC:
            ser = context.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
            ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "Public key", convert_size(len(ser))])
  
        ser = context.serialize(save_public_key=False, save_secret_key=True, save_galois_keys=False, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "Secret key",  convert_size(len(ser))])
                                    
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=True, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "Galois keys",  convert_size(len(ser))])
                                                                      
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=True)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "Relin keys",  convert_size(len(ser))])
  
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "none", convert_size(len(ser))])
        
    for (poly_mod, coeff_mod_bit_sizes) in [
        (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 40]),
        (8192, [40, 20, 40]),
        (4096, [40, 20, 40]),
        (4096, [30, 20, 30]),
        (4096, [20, 20, 20]),
        (4096, [19, 19, 19]),
        (4096, [18, 18, 18]),
        (2048, [20, 20]),
    ]:
        # BFV encrypts integers instead of decimal numbers
        context = ts.context(
            scheme=ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_mod,
            plain_modulus=786433,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        
        context.generate_galois_keys()
        context.generate_relin_keys()
        
        ser = context.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "all", convert_size(len(ser))])
        
        if enc_type is ts.ENCRYPTION_TYPE.ASYMMETRIC:
            ser = context.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
            ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "Public key", convert_size(len(ser))])
  
        ser = context.serialize(save_public_key=False, save_secret_key=True, save_galois_keys=False, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "Secret key",  convert_size(len(ser))])
                                    
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=True, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "Galois keys",  convert_size(len(ser))])
                                                                      
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=True)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "Relin keys",  convert_size(len(ser))])
  
        ser = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
        ctx_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "none", convert_size(len(ser))])
        

# display(HTML(tabulate.tabulate(ctx_size_benchmarks, tablefmt='html')))
print(tabulate.tabulate(ctx_size_benchmarks, headers="firstrow", tablefmt="grid"))

# measures the size of the encrypted data (ciphertext)

# Generate some sample data
data = [random.uniform(-10, 10) for _ in range(10 ** 3)]
network_data = pickle.dumps(data)
print("Plain data size in bytes {}".format(convert_size(len(network_data)))) # Measure the original size

enc_type = ts.ENCRYPTION_TYPE.ASYMMETRIC
ct_size_benchmarks = [["Encryption Type", "Scheme Type", "Polynomial modulus", "Coefficient modulus sizes", "Precision", "Ciphertext serialized size", "Encryption increase ratio"]]

# This loops through many different encryption settings
for (poly_mod, coeff_mod_bit_sizes, prec) in [
    (8192, [60, 40, 60], 40),
    (8192, [40, 21, 21, 21, 21, 21, 21, 40], 40),
    (8192, [40, 21, 21, 21, 21, 21, 21, 40], 21),
    (8192, [40, 20, 40], 40),
    (8192, [20, 20, 20], 38),
    (8192, [60, 60], 38),
    (8192, [40, 40], 38),
    (8192, [17, 17], 15),
    (4096, [40, 20, 40], 40),
    (4096, [30, 20, 30], 40),
    (4096, [20, 20, 20], 38),
    (4096, [19, 19, 19], 35),
    (4096, [18, 18, 18], 33),
    (4096, [30, 30], 25),
    (4096, [25, 25], 20),
    (4096, [18, 18], 16),
    (4096, [17, 17], 15),
    (2048, [20, 20], 18),
    (2048, [18, 18], 16),
    (2048, [16, 16], 14),
]:
    # Create an encryption context, creates the encryption environment 
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        encryption_type=enc_type,
    )
    # Set the scaling factor
    scale = 2 ** prec # larger scale = higher percision = larger ciphertext = more computation
    ckks_vec = ts.ckks_vector(context, data, scale)
 
    enc_network_data = ckks_vec.serialize() # converts the encrypted vector into bytes
    ct_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.CKKS], poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec),convert_size(len(enc_network_data)), round(len(enc_network_data) / len(network_data), 2)])

# repeat for BFV!    
for (poly_mod, coeff_mod_bit_sizes) in [
    (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
    (8192, [40, 21, 21, 21, 21, 21, 40]),
    (8192, [40, 21, 21, 21, 21, 40]),
    (8192, [40, 21, 21, 21, 40]),
    (8192, [40, 21, 21, 40]),
    (8192, [40, 20, 40]),
    (4096, [40, 20, 40]),
    (4096, [30, 20, 30]),
    (4096, [20, 20, 20]),
    (4096, [19, 19, 19]),
    (4096, [18, 18, 18]),
    (2048, [20, 20]),
]:
    context = ts.context(
        scheme=ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_mod,
        plain_modulus=786433,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        encryption_type=enc_type,
    )
    vec = ts.bfv_vector(context, data)
    enc_network_data = vec.serialize()
    ct_size_benchmarks.append([enc_type_str[enc_type], scheme_str[ts.SCHEME_TYPE.BFV], poly_mod, coeff_mod_bit_sizes, "-",convert_size(len(enc_network_data)), round(len(enc_network_data) / len(network_data), 2)])

    
# display(HTML(tabulate.tabulate(ct_size_benchmarks, tablefmt='html')))
print(tabulate.tabulate(ct_size_benchmarks, headers="firstrow", tablefmt="grid"))

data = [ random.random()]

enc_type = ts.ENCRYPTION_TYPE.ASYMMETRIC
ct_size_benchmarks = [["Value range", "Polynomial modulus", "Coefficient modulus sizes", "Precision", "Operation", "Status"]]


for data_pow in [-1, 0, 1, 5, 11, 21, 41, 51]:
    data = [ random.uniform(2 ** data_pow, 2 ** (data_pow + 1))]
    for (poly_mod, coeff_mod_bit_sizes, prec) in [
        (8192, [60, 40, 60], 40),
        (8192, [40, 21, 21, 21, 21, 21, 21, 40], 40),
        (8192, [40, 21, 21, 21, 21, 21, 21, 40], 21),
        (8192, [40, 20, 40], 40),
        (8192, [20, 20, 20], 38),
        (8192, [60, 60], 38),
        (8192, [40, 40], 38),
        (8192, [17, 17], 15),
        (4096, [40, 20, 40], 40),
        (4096, [30, 20, 30], 40),
        (4096, [20, 20, 20], 38),
        (4096, [19, 19, 19], 35),
        (4096, [18, 18, 18], 33),
        (4096, [30, 30], 25),
        (4096, [25, 25], 20),
        (4096, [18, 18], 16),
        (4096, [17, 17], 15),
        (2048, [20, 20], 18),
        (2048, [18, 18], 16),
        (2048, [16, 16], 14),
    ]:
        val_str = "[2^{} - 2^{}]".format(data_pow, data_pow + 1)
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        scale = 2 ** prec
        try:
            ckks_vec = ts.ckks_vector(context, data, scale)
        except BaseException as e:
            ct_size_benchmarks.append([val_str, poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec), "encrypt", "encryption failed"])
            continue
    
        decrypted = decrypt(ckks_vec)
        for dec_prec in reversed(range(prec)):
            if pytest.approx(decrypted, abs=2 ** -dec_prec) == data:
                ct_size_benchmarks.append([val_str, poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec), "encrypt", "decryption prec 2 ** {}".format(-dec_prec)])
                break
        ckks_sum = ckks_vec + ckks_vec
        decrypted = decrypt(ckks_sum)
        for dec_prec in reversed(range(prec)):
            if pytest.approx(decrypted, abs=2 ** -dec_prec) == [data[0] + data[0]]:
                ct_size_benchmarks.append([val_str, poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec), "sum", "decryption prec 2 ** {}".format(-dec_prec)])
                break
                

# We add more depth for the multiplication scenario
for data_pow in [-1, 0, 1, 5, 11, 21, 41, 51]:
    data = [ random.uniform(2 ** data_pow, 2 ** (data_pow + 1))]
    for (poly_mod, coeff_mod_bit_sizes, prec) in [
        (8192, [60, 40, 40, 60], 40),
        (8192, [40, 21, 21, 40], 40),
        (8192, [40, 21, 21, 40], 21),
        (8192, [40, 20, 20, 40], 40),
        (8192, [20, 20, 20], 38),
        (4096, [40, 20, 40], 40),
        (4096, [30, 20, 30], 40),
        (4096, [20, 20, 20], 38),
        (4096, [19, 19, 19], 35),
        (4096, [18, 18, 18], 33),
        (4096, [30, 30, 30], 25),
        (4096, [25, 25, 25], 20),
        (4096, [18, 18, 18], 16),
        (2048, [18, 18, 18], 16),
    ]:
        val_str = "[2^{} - 2^{}]".format(data_pow, data_pow + 1)
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        scale = 2 ** prec
        try:
            ckks_vec = ts.ckks_vector(context, data, scale)
        except BaseException as e:
            continue
                
        try:
            ckks_mul = ckks_vec * ckks_vec
        except:
            ct_size_benchmarks.append([val_str, poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec), "mul", "failed"])
            continue
        decrypted = decrypt(ckks_mul)
        for dec_prec in reversed(range(prec)):
            if pytest.approx(decrypted, abs=2 ** -dec_prec) == [data[0] * data[0]]:
                ct_size_benchmarks.append([val_str, poly_mod, coeff_mod_bit_sizes, "2**{}".format(prec), "mul", "decryption prec 2 ** {}".format(-dec_prec)])
                break
                
#display(HTML(tabulate.tabulate(ct_size_benchmarks, tablefmt='html')))
print(tabulate.tabulate(ct_size_benchmarks, headers="firstrow", tablefmt="grid"))