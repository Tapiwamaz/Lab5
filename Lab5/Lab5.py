import numpy as np
import math
import matplotlib.pyplot as plt

dist1X = np.random.normal(1,1,34)
dist1Y = np.random.normal(-1,1,34)

dist1Matrix = np.ones((len(dist1X),3))
for i in range(len(dist1X)):
    for j in range(1,3):
        if j ==1:
            dist1Matrix[i][j] = dist1X[i] 
        else:    
            dist1Matrix[i][j] = dist1Y[i]

dist1MatrixTraining = dist1Matrix[:20]            

dist1XTraining = dist1X[:20]
dist1YTraining = dist1Y[:20]

dist1XTest = dist1X[20:]
dist1YTest = dist1Y[20:]

dist2X = np.random.normal(-1,1,34)
dist2Y = np.random.normal(1,1,34)

dist2Matrix = np.ones((len(dist2X),3))
for i in range(len(dist2X)):
    for j in range(1,3):
        if j ==1:
            dist2Matrix[i][j] = dist2X[i] 
        else:    
            dist2Matrix[i][j] = dist2Y[i]

dist2MatrixTraining = dist2Matrix[:20]

dist2XTraining = dist2X[:20]
dist2YTraining = dist2Y[:20]

dist2XTest = dist2X[20:]
dist2YTest = dist2Y[20:]

theta = np.random.uniform(-0.5,0.5,3)

def createLineCoords(parameters, inputs ):
    coords = np.array([])
    for i in range(len(inputs)):
        y = (-parameters[1]*inputs[i] - parameters[0])/parameters[2] 
        coords = np.append(coords,y) 
    return coords    

arr = np.array([-10,10])
out = createLineCoords(theta,arr)


graph = plt.scatter(dist1XTraining,dist1YTraining,color="black")
graph = plt.scatter(dist2XTraining,dist2YTraining,color="red")
graph = plt.plot(arr,out,marker="o")
plt.xlim(-5,5)
plt.ylim(-10,10)
plt.show()


# function giving the probality of a coordinate being in a class 1
def logisticFunction(parameters,coord):
    xValue = np.dot(parameters,coord)
    result = 1 / (1 + pow(math.e,-xValue) )
    return result

print("Class of (0,5): ", round(logisticFunction(theta,np.array([1,0,5]))))

def error(parameters, dataPoints, classNumber):
    total = 0
    for point in dataPoints:
        total += classNumber * math.log10((logisticFunction(parameters,point))) + (1 -classNumber)* math.log10(1-logisticFunction(parameters,point))  
    return -total

print("Error Dist1 Class 0: ",error(theta,dist1MatrixTraining,0))
print("Error Dist1 Class 1: ",error(theta,dist1MatrixTraining,1))
print("Error Dist2 Class 0: ",error(theta,dist2MatrixTraining,0))
print("Error Dist2 Class 1: ",error(theta,dist2MatrixTraining,1))

# Distribution 1 is class 0
# Distribution 2 is class 1


confusion = np.zeros((2,2))
def confusionMatrix(parameters,dataPoints,classNumber,confusionMatrix):
    for point in dataPoints:
        prediction = round(logisticFunction(parameters,point))
        if prediction == classNumber:
            confusionMatrix[classNumber][classNumber] += 1
        else:
            confusionMatrix[prediction][classNumber] +=1
    return confusionMatrix

confusion = confusionMatrix(theta,dist1MatrixTraining,0,confusion)
confusion = confusionMatrix(theta,dist2MatrixTraining,1,confusion)
print(confusion)

