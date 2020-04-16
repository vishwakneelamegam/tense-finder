#used for math 
import numpy as np
#used to convert words into vectors
from tagger import *

#sigmoid derived
def sig_der(value):
    return value * (1 - value)

#sigmoid function
def sig(value):
    return 1 / (1 + np.exp(-value))

#training array input
train_ip = np.array([[1,1,0,0,1,0,1],
                     [0,0,0,1,1,0,1],
                     [0,0,0,1,1,1,1],
                     [0,1,0,0,1,1,1],
                     [0,1,1,1,0,0,1],
                     [0,1,1,0,0,1,1],
                     [0,1,1,1,0,1,1]])

#training array output
train_op = np.array([[1,1,1,1,0,0,0]]).T                 

#assigning random value
np.random.seed(1)                    
syn_weight = 2 * np.random.random((7,1)) - 1

#training 
for i in range(8000):
    ip = train_ip
    op = sig(np.dot(ip,syn_weight))
    error = train_op - op
    adj = error * sig_der(op)
    syn_weight += np.dot(ip.T , adj)

while True:
    ip = input("STRING : ")
    t = tag(ip)
    output = (sig(syn_weight[0][0] * t[0] + syn_weight[1][0] * t[1] + syn_weight[2][0] * t[2] + syn_weight[3][0] * t[3] + syn_weight[4][0] * t[4] + syn_weight[5][0] * t[5] + syn_weight[6][0] * t[6]))
    if output > 0.5:
        print("PAST TENSE")
    else:
        print("PRESENT TENSE")
