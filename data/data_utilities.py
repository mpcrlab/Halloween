from scipy.io import loadmat


def load_data():

    data = loadmat(file_name='./data/halloween_data.mat')
    x_train = data['x_train']
    y_train = data['y_train']
    x_valid = data['x_valid']
    y_valid = data['y_valid']

    return x_train, y_train, x_valid, y_valid
