
#激活函数
net_arch = [
{'layer' : 'dense','input_dim':6, 'output_dim':4, 'activation' : 'relu'},
{'layer' : 'dense','input_dim':6, 'output_dim':4, 'activation' : 'relu'},
{'layer' : 'dense','input_dim':6, 'output_dim':4, 'activation' : 'relu'}
]

#正向，反向激活函数及导数
def sigmoid(x):

    return 1/(1+np.exp(x))

def sigmoid_back(x, dx):

    return dx * sigmoid(x) * (1-sigmoid(x))

def relu(x):

    return np.maximum(0,x)

def relu_back(x,dx):

    dx = np.array(dx)
    dx[x <= 0] = 0
    return dx

#全连接层
class DenseLayer:

    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initialize_weights()

#初始化权层矩阵
def initialize_weights(self):

    self.w = np.random.uniform(0,1,size=(self.input_dim, self.output_dim))
    self.b = np.random.uniform(0,1,size=(self.output_dim,))

#编码前向传播过程
def dense_layer_forward(self,prevX):

    crr_non_activated_output = np.dot(self.w, prevX) + self.b
    if self.activation is 'relu':
        activation_func = relu

    elif self.activation is 'sigmoid':
        activation_func = sigmoid

    else:
        raise Exception('輸入的什么玩意，报个错吧')
    self.crr_activated_output = activation_func(crr_non_activated_output)
    self.crr_non_activated_output = crr_non_activated_output
    return crr_non_activated_output

#反向函数
def dense_layer_backward(self,d_crrx, prevX):

    previous_x_dim = prevX.shape[1]
    if self.activation is 'sigmoid':
        backward_activation_func = sigmoid_back

    elif self.activation is 'relu':
        backward_activation_func = relu_back

    else:
        raise Exception('输入的什么玩意，看不懂，报个错吧')

    d_rcc_activated_output = backward_activation_func(d_rccx,self.crr_activated_output)

    dw = np.dot(d_crr_non_activated_output, prevX.T)/ previous_x_dim
    db = np.sum(d_crr_non_activated_output, axis = 1, Keepdims=True)/previous_x_dim
    d_prevX = np.dot(dw.T, d_crr_non_activated_output)
    return d_prevX, dw, db

#权重更新
def update_grads(self,dw , db, learning_rate=0.1):

    self.w -= dw * learning_rate
    self.b -= db * learning_rate

#整合所有特征层
class JianDanNN:

    def __init__(self,nn_arch):
        self.nn_arch = nn_arch
        self.layers = []
        self.build_layers()

    def build_layers(self):
        for i, i_config in enumerate(self.nn_arch):
            layer = self.build_layer(
                type = i_config['layer'],
                input_dim = int(i_config['input_dim']),
                output_dim = int(i_config['output_dim']),
                activation= i_config['activation']
            )
            self.layers.append(layer)

    def forward(self,x):
            crrx = x
            for layer in self.layers:
                prevX = crrx
                crrx = layer.dense_layer_forward(prevX)

            return crrx


    def backward(self,estimated_output,Y):

        d_prevX = -(np.divide(Y,estimated_output) - np.divide(1-Y,1-estimated_output))

        for prev_idx in range(len(self.layers), 0, -1):

          d_crrx = d_prevX
          crr_idx = prev_idx -1
          prev_layer = self.layers[prev_idx]
          crr_layer = self.layers[crr_idx]
          prevX = prev_layer.crr_non_activated_output
          d_prevX, dw, db = crr_layer.dense_layer_backward(d_crrx,prevX)
          crr_layer.update_grads(dw,db)

    def get_batch_data(self,inputs,labels,size):
        if len(inputs)> 0:
            random_indices = np.random.randint(0,inputs.shape[0], min(self.batch_size,input.shape[0]))
            input_samples = inputs[random_indices]
            label_samples = labels[random_indices]
            np.delete(inputs, random_indices,axis=0)
            np.delete(labels, random_indices, axis=0)
            return input_samples, label_samples
        else:
            return [], []

    def train(self, inputs,labels,max_iters,batch_size=64):
        for i in range(max_iters):
            x_copy = copy.copy(inputs)
            y_copy = copy.copy(labels)
            input_samples, label_samples = self.get_batch_data(x_copy,y_copy,batch_size)
            while len(input_samples)> 0 :
                E = self.forward(input_samples)
                self.backward(E,label_samples)