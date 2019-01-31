import tensorflow as tf
import pandas as pd
import matplotlib as plt

data_path = r'F:\Python\Coursera_ML\machine-learning-ex1\ex1\mydata.txt'
training_epochs = 1000 #iterations for error optimization
n_dim = 1

class Linear_Regression():
    def __init__(self):
        #tain and test data
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        #model lables
        self.a = tf.Variable([-1.0],tf.float32) #Theta1 (factor)
        self.b = tf.Variable([1.0],tf.float32) #Theta2
        self.x = tf.placeholder(tf.float32) # x - gradient (prediction)
        self.y = tf.placeholder(tf.float32) # y - Actual Value

        #functions
        self.loss = None
        self.loss_arr = []

        #session
        self.sess = None

    #splits and read txt(csv) file to test and train arrays
    def read_into_arrays(self,path=data_path,ratio=0.8):
        data = pd.read_csv(path, header=None)
        x_total = list(data[0])
        y_total = list(data[2])
        self.x_train = x_total[0:int(ratio*len(x_total))]  #80% reserved to Train data
        self.x_test  = x_total[int(ratio*len(x_total)):len(x_total)] #20% for test
        self.y_train = y_total[0:int(ratio*len(y_total))]  #80% reserved to Train data
        self.y_test  = y_total[int(ratio*len(y_total)):len(y_total)] #20% for test
            
    #linear model implementation
    def model_implementation(self):
        linear_model = self.a*self.x + self.b #representation of a linear regression model
        squared_delta = tf.square(linear_model-self.y) #squared loss
        self.loss = tf.reduce_mean(squared_delta) #total sum of loss for m inputs

    #optimizer based on gradient descent algorithm
    def loss_optimized_factors(self,learning_rate=0.00000001,training_epochs=500):        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()    
        self.sess.run(init)
        for _ in range(training_epochs):
            self.sess.run(train,{self.x:self.x_train,self.y:self.y_train}) 
            self.loss_arr.append(self.loss)
        return self.a,self.b,self.loss

    def plot_loss(self):
        iter = list(range(1,501))
        plt.pyplot.plot(iter,self.loss_arr)
        plt.pyplot.show()    

linear_reg = Linear_Regression()
linear_reg.read_into_arrays()
linear_reg.model_implementation()
theta1, theta2, loss = linear_reg.loss_optimized_factors()
theta1 = linear_reg.sess.run([theta1])[0]
theta2 = linear_reg.sess.run([theta2])[0]
linear_reg.plot_loss()
linear_reg.sess.close()

predection_vs_actual = []
for ind in range(len(linear_reg.x_test)):
    prediceted = theta1*linear_reg.x_test[ind] + theta2 
    actual = linear_reg.y_test[ind]
    predection_vs_actual.append([prediceted[0],actual])

print(predection_vs_actual)