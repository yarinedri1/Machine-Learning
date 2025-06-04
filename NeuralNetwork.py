import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation Functions(sigmoid)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Loss Function (Binary Cross-Entropy)

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Neural Network Parameters Initialization

def initialize_parameters(n_input, n_hidden):
    W1 = np.random.randn(n_input, n_hidden) * 0.01  # Initialize weights of input layer(1) and hidden layer(2) with small random values
    b1 = np.random.randn(1, n_hidden) * 0.01        # Initialize biases of input layer(1) and hidden layer(2) with small random values
    W2 = np.random.randn(n_hidden, 1) * 0.01        
    b2 = np.random.randn(1, 1) * 0.01               
    return W1, b1, W2, b2

### Using BackPropagation mathod to calculate gradients in efficient way ###

# Forward pass

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)              # input layer output => hidden layer input
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)              # hidden layer output => output layer input
    return A1, A2


# Backward Propagation

def backward(X, y, A1, A2, W2):
    m = y.shape[0]         #number of training samples

    # gradient of loss function w.r.t NN prameters 
    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / m        
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2                    

# Training Function

def train(samples_train, labels_train, samples_test, labels_test, size_hidden, epochs, learning_rate):
    size_input = samples_train.shape[1]              # Number of features in the input layer
    W1, b1, W2, b2 = initialize_parameters(size_input, size_hidden)

    loss_train = []
    loss_test = []
    checkpoints = []

    for epoch in range(1, epochs + 1):
        A1_train, A2_train = forward(samples_train, W1, b1, W2, b2)
        loss = compute_loss(labels_train, A2_train)

        dW1, db1, dW2, db2 = backward(samples_train, labels_train, A1_train, A2_train, W2)

        # Gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1         ## learning rate is the "step size" (u) for updating the weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Record every 50 epochs
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            A1_test, A2_test = forward(samples_test, W1, b1, W2, b2)
            loss_t = compute_loss(labels_test, A2_test)
            loss_train.append(loss)
            loss_test.append(loss_t)
            checkpoints.append(epoch)
            print(f"Epoch {epoch:4d} - Train Loss: {loss:.4f} - Test Loss: {loss_t:.4f}")

    return W1, b1, W2, b2, loss_train, loss_test, checkpoints


# Prediction

def predict(Input, W1, b1, W2, b2):
    _, A2 = forward(Input, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)


### Main ###

def main():
    # Load and prepare data
    data = load_breast_cancer()
    DataSet = data.data
    Labels = data.target.reshape(-1, 1)

    # Normalize
    scaler = StandardScaler()            
    DataSet = scaler.fit_transform(DataSet)

    # Split
    Samples_train, Samples_test, labels_train, labels_test = train_test_split(DataSet, Labels, test_size=0.2, random_state=42)

    # Train model
    opt_W1, opt_b1, opt_W2, opt_b2, loss_train, loss_test, checkpoints = train(
        Samples_train, labels_train, Samples_test, labels_test,
        size_hidden=50,
        epochs=1000,
        learning_rate=0.1
    )

    # Test accuracy
    y_pred = predict(Samples_test, opt_W1, opt_b1, opt_W2, opt_b2)
    accuracy = np.mean(y_pred == labels_test)         #comparing predicted labels with ground truth labels
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Plot training and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, loss_train, label="Train Loss", marker='o')
    plt.plot(checkpoints, loss_test, label="Test Loss", marker='s')
    plt.title("Train Loss vs. Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
