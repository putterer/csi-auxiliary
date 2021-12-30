from keras import Model
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, ReLU, LSTM, TimeDistributed

class SimpleSequentialModel(Model):
    class_count: int

    def __init__(self, use_batchnorm, class_count, input_shape):
        super().__init__()

        self.class_count = class_count
        self.use_batchnorm = use_batchnorm

        self.conv1 = Conv2D(16, kernel_size = (7, 7), activation = 'relu', input_shape = input_shape)
        self.conv2 = Conv2D(32, (5, 5), activation = 'relu')
        self.conv3 = Conv2D(32, (3, 3), activation = 'relu')
        #self.conv1 = CConv2D(16, kernel_size = (7, 7), activation = 'relu', input_shape = input_shape)
        #self.conv2 = CConv2D(32, (5, 5), activation = 'relu')
        #self.conv3 = CConv2D(16, (3, 3), activation = 'relu')
        self.pool1 = MaxPooling2D(pool_size = (2, 2))
        self.dropout1 = Dropout(0.25)

        self.flatten = Flatten()
        #self.dense1 = Dense(128, activation = 'relu')
        self.dense1 = Dense(128)
        self.batchnorm1 = BatchNormalization()
        self.activation1 = ReLU()
        self.dropout2 = Dropout(0.5)
        self.dense2 = Dense(class_count, activation = 'softmax')

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.dropout1(x, training = training)
        x = self.flatten(x)
        x = self.dense1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.dropout2(x, training = training)
        x = self.dense2(x)

        return x



class SequentialCNNBlock(Model):
    class_count: int

    def __init__(self, class_count, input_shape):
        super().__init__()

        self.class_count = class_count

        self.conv1 = Conv2D(16, kernel_size = (11, 11), activation = 'relu', input_shape = input_shape)
        self.conv2 = Conv2D(32, (7, 7), activation = 'relu')
        self.conv3 = Conv2D(16, (5, 5), activation = 'relu')
        self.pool1 = MaxPooling2D(pool_size = (3, 3))
        self.dropout1 = Dropout(0.25)

        self.flatten = Flatten()

        self.output_dimension = 128
        self.dense1 = Dense(self.output_dimension)
        #self.batchnorm1 = BatchNormalization()
        #self.activation1 = ReLU()
        self.dropout2 = Dropout(0.5)



    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.dropout1(x, training = training)
        x = self.flatten(x)

        x = self.dense1(x)
        #x = self.batchnorm1(x)
        #x = self.activation1(x)
        x = self.dropout2(x, training = training)

        return x

    def compute_output_shape(self, input_shape):
        # no padding, reduced by kernel size - 1
        # dim1 = input_shape[1] - 6 - 4 - 2
        # dim2 = input_shape[2] - 6 - 4 - 2

        # return (input_shape[0], dim1 * dim2 * input_shape[3]) # samples x flattened
        return (input_shape[0], self.output_dimension)


class SimpleRecursiveModel(Model):
    class_count: int

    def __init__(self, class_count, input_shape):
        super().__init__()

        self.class_count = class_count

        self.time_distributed_sequential_block = TimeDistributed(SequentialCNNBlock(class_count, input_shape)) # input shape remove one dimension
        self.lstm = LSTM(128)
        self.dense = Dense(class_count, activation="softmax")


    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.time_distributed_sequential_block(x)
        x = self.lstm(x)
        x = self.dense(x)

        return x



# model = Sequential()
# model.add(Conv2D(32, kernel_size = (7, 7), activation = 'relu', input_shape = input_shape))
# #model.add(Conv2D(64, kernel_size = (5, 5), activation = 'relu'))
# model.add(Conv2D(64, (5, 5), activation = 'relu'))
# #model.add(Conv2D(64, (3, 3), activation = 'relu'))
# model.add(Conv2D(64, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(classCount, activation = 'softmax'))