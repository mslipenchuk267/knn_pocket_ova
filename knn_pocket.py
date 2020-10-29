import csv
import numpy as np

def compute_accuracy(test_y, pred_y):    
    # TO-DO: add your code here
    return None

def test_knn(train_x, train_y, test_x, num_nn):    
    # TO-DO: add your code here
    return None

def test_pocket(w, test_x):    
    # TO-DO: add your code here
    return None
def train_pocket(train_x, train_y, num_iters):    
    # TO-DO: add your code here
    return None
def get_id():    
    return 'tuf91673'

def main():
    datasetPath = './letter-recognition.data'
    listClasses=[]
    listAttrs = []
    with open(datasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            listClasses.append(row[0])
            listAttrs.append(list(map(float, row[1:])))
    
    # Generate mapping from class name (i.e the letter) to integer IDs
    mapCls2Int = dict([y, x] for x, y in enumerate(sorted(set(listClasses))))

    # Store the dataset with numpy array
    dataX = np.array(listAttrs)
    dataY = np.array([mapCls2Int[cls] for cls in listClasses])

    # Split the dataset into training set and testing set
    numTrainingExamples = 15000
    trainX = dataX[:numTrainingExamples, :]
    trainY = dataY[:numTrainingExamples]
    testX = dataX[numTrainingExamples:, :]
    testY = dataY[numTrainingExamples:]

    # TO-DO: Add code here

    print("Skeleton is completed")

if __name__ == "__main__":
    main()