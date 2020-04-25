#numpy for calculation
import numpy as np
from csv import reader
# read csv file as a list of lists
data = []

with open('grammarToVector1.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    data = list(csv_reader)
refer = data.pop(0)
refer.pop(-1)

#sigmoid function ranges squash value x between 0 and 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid_p is for derived sigmoid 0.25 to 0
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#neuralNetwork function
def neuralNetwork(ipList,weightList,b):
    result = 0
    for i in range(0,52):
        result += float(ipList[i]) * weightList[i]
    return result + b

#trainData to train data
def trainData():
    output = []
    weights = []
    cost_w = []
    for _ in range(0,52):
        weights.append(np.random.rand())
        cost_w.append(0)
        output.append(0)
    b  = np.random.randn()
    #initialize iteration and learning rate
    iteration = 100000
    rate = 0.1
    try:
        for i in range(iteration):
            #random integer of size of data
            randomInput = np.random.randint(len(data))
            #predicted value
            op      = neuralNetwork(data[randomInput],weights,b)
            predict = sigmoid(op)
            #loss function
            cost_predict = 2.0 * (float(predict) - float(data[randomInput][52]))
            #print(cost_predict)
            predict_op   = sigmoid_p(op)
            #print("predict op : " + str(predict_op))
            for i in range(0,52):
                cost_w[i]      = float(cost_predict) * float(predict_op) * float(data[randomInput][i])
            cost_b       = float(cost_predict) * float(predict_op) * 1.0
            for i in range(0,52):
                weights[i] = float(weights[i]) - float(rate) * float(cost_w[i])
            b  = float(b)  - rate * float(cost_b)
        return weights,b
    except Exception as e:
        return str(e)

x,y = trainData()

while True:
    inp = input("STRING : ")
    splits = inp.split(" ")
    cal = []
    for data in refer:
        if data in splits:
            cal.append(1)
        else:
            cal.append(0)
    print(sigmoid(neuralNetwork(cal,x,y)))
