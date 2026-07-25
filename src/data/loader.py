from tensorflow.keras.datasets import mnist



def load_mnist():
    return mnist.load_data()


def load_emnist():
    """
    EMNIST Letters Dataset
    Returns:
        (x_train,y_train),(x_test,y_test)
    """
    return emnist.load_data(type="letters")