import csv
import numpy as np

def compute_accuracy(test_y, pred_y):
    """Compute the accuracy of the predicted test labels against the ground truth test labels

    Args:
        test_y (np.array): ground truth labels of the test_x data set (num_test,)
        pred_y (np.array): predicted labels of the test_x data set (num_test,)

    Returns:
        float: acc = float between 0 and 1 representign the classification accuracy
    """

    # TO-DO: add your code here
    return None

def test_knn(train_x, train_y, test_x, num_nn):
    """Train the kNN model with the training data set and then Predict the labels of the test set.

    Args:
        train_x (np.array): The attributes for each observation in training set (num_train, num_attrs)
        train_y (np.array): The classes for each observation in the training set (num_train,)
        test_x (np.array): an array of the attributes for each class (num_test, num_attrs)
        num_nn (int): the number of nearest neighbors to be used for classification

    Returns:
        np.array: pred_y = predicted labels of the test_x data set (num_test,)
    """

    # TO-DO: add your code here
    return None

def test_pocket(w, test_x):
    """Predict the labels of the training data set using the weights generated in train_pocket()

    Args:
        w (np.array): The learned perception weights
        test_x (np.array): an array of the attributes for each class (num_test, num_attrs)

    Returns:
        np.array: pred_y = predicted labels of the test_x data set (num_test,)
    """

    # TO-DO: add your code here
    return None

def train_pocket(train_x, train_y, num_iters):
    """Train the pocket algorithm with the training data set over a specified number of iterations.
        It returns the weights of the trained model.

    Args:
        train_x (np.array): The attributes for each observation in training set (num_train, num_attrs)
        train_y (np.array): [description]
        num_iters (int): The number of iterations of the algorithm

    Returns:
        np.array: w = weights generated after having trained the Pocket model over num_iters iterations
    """

    # TO-DO: add your code here
    return None

def get_id():
    """Simply return the student's Temple Accessnet ID.

    Returns:
        string: The student's Temple Accessnet ID
    """
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