

def train_test_split(data,test_size = .25):
    np.random.shuffle(data)
    test = data[int(round((len(data)*test_size))):]
    train = data[:(int(round(len(data)*test_size)))]
    test_X = test[:,:-1]
    test_y = test[:,-1:]
    train_X = train[:,:-1]
    train_y =  train[:,-1:]
    return train_X,train_y,test_X,test_y