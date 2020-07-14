import numpy as np
import copy


np.random.seed(1)


############################################################
## Part 1: Prapare dataset and the sigmoid function ########
############################################################

# prepare the dataset
# max dim of binary numbers we use (also the sequence length)
max_dim = 8
# store binary numbers in a look-up table X
X = np.unpackbits(np.array([range(2**max_dim)], dtype=np.uint8).T, axis=1)


def get_sample():
    # generate a sample
    x1_int = np.random.randint(2**(max_dim-1))
    x2_int = np.random.randint(2**(max_dim-1))
    y_int = x1_int + x2_int
    # look up the binary numbers
    x1 = X[x1_int]
    x2 = X[x2_int]
    y = X[y_int]

    return x1, x2, y, x1_int, x2_int, y_int


# let's see some examples
print('shape of X', X.shape)
x1, x2, y, x1_int, x2_int, y_int = get_sample()
print('')
print('x1:   '+str(x1)+'  '+str(x1_int))
print('x2:   '+str(x2)+'  '+str(x2_int))
print('y :   '+str(y)+'  '+str(y_int))
print('-'*20)

# sigmoid function
# output has the same dim with x


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# test your sigmoid
print('sigmoid(2) = %f, the result should equal 0.880797' % (sigmoid(2.0)))

# grad of input


def sigmoid_grad_input(output):
    input_grad = output * (1 - output)
    return input_grad


# test your sigmoid grad
print('sigmoid_grad_input(0.5) = %f, the result should equal 0.25' %
      (sigmoid_grad_input(0.5)))
print('-'*20)


############################################################
## Part 2: Initialize the rnn                       ########
############################################################

# learning rate
lr = 0.05
input_dim = 2
output_dim = 1
hidden_dim = 128

# initialize neural network weights
W1 = 2*np.random.random((input_dim, hidden_dim)) - 1
W2 = 2*np.random.random((hidden_dim, output_dim)) - 1
Wh = 2*np.random.random((hidden_dim, hidden_dim)) - 1
# grad for W
W1_grad = np.zeros_like(W1)
W2_grad = np.zeros_like(W2)
Wh_grad = np.zeros_like(Wh)


############################################################
## Part 3: Training the rnn                         ########
############################################################
# in each iter we first
# 1) forward the max_dim steps, then
# 2) backward the grad for each W and update W
for i in range(5000):

    # get a sample
    x1, x2, y, x1_int, x2_int, y_int = get_sample()

    # value of the hidden layer
    hiddens = list()
    hiddens.append(np.zeros(hidden_dim))
    # value of the output
    outs = list()

    # prediction of the rnn
    loss = 0
    pred = np.zeros_like(y)

    # forward
    for j in range(max_dim):

        # get input and output
        position = max_dim-j-1
        x_t = np.array([[x1[position], x2[position]]])
        y_t = np.array([[y[position]]]).T
        hidden_t = sigmoid(np.dot(x_t, W1) + np.dot(hiddens[j], Wh))
        # store hidden_t
        hiddens.append(copy.deepcopy(hidden_t))
        out_t = sigmoid(np.dot(hidden_t, W2))
        # store hidden_t
        outs.append(copy.deepcopy(out_t))
        # sum up the loss
        loss += -(y_t*np.log(out_t) + (1-y_t)*np.log(1-out_t))[0][0]
        # make the prediction, according to out_t
        if out_t > 0.5:
            pred[position] = 1
        else:
            pred[position] = 0

    # store the grad of t+1 hidden layer
    hidden_next_grad = np.zeros(hidden_dim)
    # backward
    for j in range(max_dim):

        # get input and output
        x_t = np.array([[x1[j], x2[j]]])
        y_t = np.array([[y[j]]]).T
        out_t = outs[-j-1]
        hidden_t = hiddens[-j-1]
        # t-1 hidden layer
        hidden_t_grad = np.transpose(W2) * (out_t - y_t) + np.dot(hidden_next_grad, np.dot(
            np.diag(sigmoid_grad_input(hiddens[-j]).reshape(-1)), np.transpose(Wh)))
        W2_grad += (out_t - y_t) * np.transpose(hidden_t)
        Wh_grad += np.dot(np.dot(hiddens[-j-2].reshape(-1, 1), hidden_t_grad),
                          np.diag(sigmoid_grad_input(hidden_t).reshape(-1)))
        W1_grad += np.dot(np.dot(np.transpose(x_t), hidden_t_grad),
                          np.diag(sigmoid_grad_input(hidden_t).reshape(-1)))

        # store the grad of t hidden layer
        hidden_next_grad = hidden_t_grad

    # update W, lr is the learning rate
    W1 -= W1_grad * lr
    W2 -= W2_grad * lr
    Wh -= Wh_grad * lr

    # reset the grad of W
    W1_grad *= 0
    W2_grad *= 0
    Wh_grad *= 0

    # print out progress
    if(i % 100 == 0):
        print('i:'+str(i))
        print("loss:" + str(loss))
        print('x1:   '+str(x1)+'  '+str(x1_int))
        print('x2:   '+str(x2)+'  '+str(x2_int))
        print('y :   '+str(y)+'  '+str(y_int))
        pred_int = 0
        for index, x in enumerate(reversed(pred)):
            pred_int += x*(2**index)
        print('pred: '+str(pred)+'  '+str(pred_int))
        print('------------')
