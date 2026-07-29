from pathlib import Path
import torch
import tenseal as ts
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

log_file = Path("training.log")
log_file.write_text("")


def log(message):
    print(message)
    with log_file.open("a") as f:
        f.write(str(message) + "\n")


# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------

torch.random.manual_seed(73)
random.seed(73)

data = pd.read_csv("./data/framingham.csv")

log(data.columns.tolist())


# -----------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)

    delim = int(len(x) * test_ratio)

    test_idxs = idxs[:delim]
    train_idxs = idxs[delim:]

    return (
        x[train_idxs],
        y[train_idxs],
        x[test_idxs],
        y[test_idxs],
    )


# -----------------------------------------------------------------------------
# Heart Disease Dataset
# -----------------------------------------------------------------------------

def heart_disease_data():
    data = pd.read_csv("./data/framingham.csv")

    # Remove missing values
    data = data.dropna()

    # Drop unused features
    data = data.drop(
        columns=[
            "education",
            "currentSmoker",
            "BPMeds",
            "diabetes",
            "diaBP",
            "BMI",
        ]
    )

    # Balance the classes
    min_size = data["TenYearCHD"].value_counts().min()

    data_0 = data[data["TenYearCHD"] == 0].sample(
        min_size,
        random_state=73,
    )

    data_1 = data[data["TenYearCHD"] == 1].sample(
        min_size,
        random_state=73,
    )

    data = pd.concat([data_0, data_1]).reset_index(drop=True)

    # Labels
    y = torch.tensor(
        data["TenYearCHD"].values
    ).float().unsqueeze(1)

    # Remove label
    data = data.drop(columns=["TenYearCHD"])

    # Standardize
    data = (data - data.mean()) / data.std()

    x = torch.tensor(data.values).float()

    return split_train_test(x, y)


# -----------------------------------------------------------------------------
# Random Dataset (Optional)
# -----------------------------------------------------------------------------

def random_data(m=1024, n=2):
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)

    y_train = (
        (x_train[:, 0] >= x_train[:, 1])
        .float()
        .unsqueeze(0)
        .t()
    )

    y_test = (
        (x_test[:, 0] >= x_test[:, 1])
        .float()
        .unsqueeze(0)
        .t()
    )

    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------------
# Select Dataset
# -----------------------------------------------------------------------------

# x_train, y_train, x_test, y_test = random_data()
x_train, y_train, x_test, y_test = heart_disease_data()

log("############# Data summary #############")
log(f"x_train has shape: {x_train.shape}")
log(f"y_train has shape: {y_train.shape}")
log(f"x_test has shape: {x_test.shape}")
log(f"y_test has shape: {y_test.shape}")
log("#######################################")


# -----------------------------------------------------------------------------
# Logistic Regression
# -----------------------------------------------------------------------------

class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.lr(x))


n_features = x_train.shape[1]

model = LR(n_features)

optim = torch.optim.SGD(
    model.parameters(),
    lr=1,
)

criterion = torch.nn.BCELoss()

EPOCHS = 5


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(model, optim, criterion, x, y, epochs=EPOCHS):

    for e in range(1, epochs + 1):

        optim.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optim.step()

        log(f"Loss at epoch {e}: {loss.data}")

    return model


model = train(
    model,
    optim,
    criterion,
    x_train,
    y_train,
)


# -----------------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------------

def accuracy(model, x, y):

    out = model(x)

    correct = torch.abs(y - out) < 0.5

    return correct.float().mean()


plain_accuracy = accuracy(
    model,
    x_test,
    y_test,
)

log(f"Accuracy on plain test_set: {plain_accuracy}")

# -----------------------------------------------------------------------------
# Encrypted Evaluation
# -----------------------------------------------------------------------------

class EncryptedLR:

    def __init__(self, torch_lr):
        # TenSEAL works with Python lists instead of tensors
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()

    def forward(self, enc_x):
        # No sigmoid needed for evaluation
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -------------------------------------------------------
    # Optional model encryption
    # -------------------------------------------------------

    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()


eelr = EncryptedLR(model)

# -----------------------------------------------------------------------------
# TenSEAL Context (Evaluation)
# -----------------------------------------------------------------------------

poly_mod_degree = 4096

coeff_mod_bit_sizes = [
    40,
    20,
    40,
]

ctx_eval = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_mod_degree,
    -1,
    coeff_mod_bit_sizes,
)

ctx_eval.global_scale = 2 ** 20

ctx_eval.generate_galois_keys()

# -----------------------------------------------------------------------------
# Encrypt Test Set
# -----------------------------------------------------------------------------

t_start = time()

enc_x_test = [
    ts.ckks_vector(ctx_eval, x.tolist())
    for x in x_test
]

t_end = time()

log(
    f"Encryption of the test-set took {int(t_end - t_start)} seconds"
)

# Optional:
# eelr.encrypt(ctx_eval)

# -----------------------------------------------------------------------------
# Encrypted Evaluation Function
# -----------------------------------------------------------------------------

def encrypted_evaluation(model, enc_x_test, y_test):

    t_start = time()

    correct = 0

    for enc_x, y in zip(enc_x_test, y_test):

        # Forward pass on encrypted data
        enc_out = model(enc_x)

        # Decrypt prediction
        out = enc_out.decrypt()

        out = torch.tensor(out)

        out = torch.sigmoid(out)

        if torch.abs(out - y) < 0.5:
            correct += 1

    t_end = time()

    log(
        f"Evaluated test_set of {len(x_test)} entries "
        f"in {int(t_end - t_start)} seconds"
    )

    log(
        f"Accuracy: {correct}/{len(x_test)} = "
        f"{correct / len(x_test)}"
    )

    return correct / len(x_test)


# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------

encrypted_accuracy = encrypted_evaluation(
    eelr,
    enc_x_test,
    y_test,
)

diff_accuracy = plain_accuracy - encrypted_accuracy

log(
    f"Difference between plain and encrypted accuracies: "
    f"{diff_accuracy}"
)

if diff_accuracy < 0:
    log(
        "Oh! We got a better accuracy on the encrypted "
        "test-set! The noise was on our side..."
    )

    # -----------------------------------------------------------------------------
# Training an Encrypted Logistic Regression Model
# -----------------------------------------------------------------------------

class EncryptedLR:

    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()

        # Gradient accumulators
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = enc_out - enc_y

        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1

    def update_parameters(self):

        if self._count == 0:
            raise RuntimeError(
                "You should at least run one forward iteration"
            )

        # Regularized gradient update
        self.weight -= (
            self._delta_w * (1 / self._count)
            + self.weight * 0.05
        )

        self.bias -= (
            self._delta_b * (1 / self._count)
        )

        # Reset accumulators
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    @staticmethod
    def sigmoid(enc_x):
        # Degree-3 polynomial approximation
        # sigmoid(x) ≈ 0.5 + 0.197x - 0.004x³
        return enc_x.polyval([
            0.5,
            0.197,
            0,
            -0.004
        ])

    def plain_accuracy(self, x_test, y_test):

        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)

        out = torch.sigmoid(
            x_test.matmul(w) + b
        ).reshape(-1, 1)

        correct = torch.abs(y_test - out) < 0.5

        return correct.float().mean()

    def encrypt(self, context):
        self.weight = ts.ckks_vector(
            context,
            self.weight,
        )

        self.bias = ts.ckks_vector(
            context,
            self.bias,
        )

    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# -----------------------------------------------------------------------------
# TenSEAL Context (Training)
# -----------------------------------------------------------------------------

poly_mod_degree = 8192

coeff_mod_bit_sizes = [
    40,
    21,
    21,
    21,
    21,
    21,
    21,
    40,
]

ctx_training = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_mod_degree,
    -1,
    coeff_mod_bit_sizes,
)

ctx_training.global_scale = 2 ** 21

ctx_training.generate_galois_keys()

# -----------------------------------------------------------------------------
# Encrypt Training Data
# -----------------------------------------------------------------------------

t_start = time()

enc_x_train = [
    ts.ckks_vector(
        ctx_training,
        x.tolist(),
    )
    for x in x_train
]

enc_y_train = [
    ts.ckks_vector(
        ctx_training,
        y.tolist(),
    )
    for y in y_train
]

t_end = time()

log(
    f"Encryption of the training_set took "
    f"{int(t_end - t_start)} seconds"
)

# -----------------------------------------------------------------------------
# Distribution Utilities
# -----------------------------------------------------------------------------

normal_dist = (
    lambda x, mean, var:
    np.exp(
        -np.square(x - mean) / (2 * var)
    ) / np.sqrt(2 * np.pi * var)
)


def plot_normal_dist(
    mean,
    var,
    rmin=-10,
    rmax=10,
):
    x = np.arange(
        rmin,
        rmax,
        0.01,
    )

    y = normal_dist(
        x,
        mean,
        var,
    )

    plt.plot(x, y)


# -----------------------------------------------------------------------------
# Plain Distribution
# -----------------------------------------------------------------------------

lr = LR(n_features)

data = lr.lr(x_test)

mean, var = map(
    float,
    [
        data.mean(),
        data.std() ** 2,
    ],
)

plot_normal_dist(
    mean,
    var,
)

log("Distribution on plain data:")

plt.show()

# -----------------------------------------------------------------------------
# Encrypted Distribution
# -----------------------------------------------------------------------------

def encrypted_out_distribution(
    eelr,
    enc_x_test,
):

    w = eelr.weight
    b = eelr.bias

    data = []

    for enc_x in enc_x_test:

        enc_out = enc_x.dot(w) + b

        data.append(
            enc_out.decrypt()
        )

    data = torch.tensor(data)

    mean, var = map(
        float,
        [
            data.mean(),
            data.std() ** 2,
        ],
    )

    plot_normal_dist(
        mean,
        var,
    )

    log("Distribution on encrypted data:")

    plt.show()


eelr = EncryptedLR(lr)

eelr.encrypt(ctx_training)

encrypted_out_distribution(
    eelr,
    enc_x_train,
)
# -----------------------------------------------------------------------------
# Train Encrypted Logistic Regression
# -----------------------------------------------------------------------------

eelr = EncryptedLR(LR(n_features))

accuracy = eelr.plain_accuracy(
    x_test,
    y_test,
)

log(f"Accuracy at epoch #0 is {accuracy}")

times = []

for epoch in range(EPOCHS):

    # Encrypt model parameters
    eelr.encrypt(ctx_training)

    # -------------------------------------------------
    # Optional: inspect distribution each epoch
    # (Very slow)
    # -------------------------------------------------

    # encrypted_out_distribution(
    #     eelr,
    #     enc_x_train,
    # )

    t_start = time()

    # Forward + Backward
    for enc_x, enc_y in zip(
        enc_x_train,
        enc_y_train,
    ):

        enc_out = eelr.forward(enc_x)

        eelr.backward(
            enc_x,
            enc_out,
            enc_y,
        )

    # Update parameters
    eelr.update_parameters()

    t_end = time()

    times.append(
        t_end - t_start
    )

    # Decrypt weights so we can evaluate
    eelr.decrypt()

    accuracy = eelr.plain_accuracy(
        x_test,
        y_test,
    )

    log(
        f"Accuracy at epoch #{epoch + 1} is {accuracy}"
    )


# -----------------------------------------------------------------------------
# Training Statistics
# -----------------------------------------------------------------------------

log(
    f"\nAverage time per epoch: "
    f"{int(sum(times) / len(times))} seconds"
)

log(
    f"Final accuracy is {accuracy}"
)

diff_accuracy = (
    plain_accuracy - accuracy
)

log(
    f"Difference between plain and encrypted "
    f"accuracies: {diff_accuracy}"
)

if diff_accuracy < 0:

    log(
        "Oh! We got a better accuracy when "
        "training on encrypted data! "
        "The noise was on our side..."
    )

log(
    f"Final encrypted accuracy: {accuracy}"
)

log(
    f"Difference from plain accuracy: "
    f"{plain_accuracy - accuracy}"
)