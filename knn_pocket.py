import csv
import numpy as np
from scipy.spatial import distance
from scipy import stats
import datetime
import matplotlib.pyplot as plt


def compute_accuracy(test_y, pred_y):
    """Compute the accuracy of the predicted test labels against the ground truth test labels

    Args:
        test_y (np.array): ground truth labels of the test_x data set (num_test,)
        pred_y (np.array): predicted labels of the test_x data set (num_test,)

    Returns:
        float: acc = float between 0 and 1 representign the classification accuracy
    """

    acc = np.where(test_y == pred_y)
    acc = acc[0].shape[0] / len(test_y) * 100
    return acc


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

    # Partially Vectorized Version
    #  I vectorized computing the distances and then iteratively choose the num_nn closest neighbors
    #  and select the most common label.
    pred_y = np.empty(0)
    # Compute the euclidean distance on each testing observation against all training observations
    distances = distance.cdist(test_x, train_x, metric='euclidean')
    for i, observation_distances in enumerate(distances):
        # Return the num_nn closest neighbor indices (i.e. the values with shortest distances)
        closest_neighbors = observation_distances.argsort()[:num_nn]
        # Get the labels from train_y using the indices of the closest neighbors
        closest_labels = [train_y[closest_neighbor_index]
                          for closest_neighbor_index in closest_neighbors]
        # Select the most common label
        predicted_label = stats.mode(closest_labels)[0][0]
        print(f"Percent Done: {round(i/len(distances)*100, 3)}%", end='\r')
        pred_y = np.append(pred_y, predicted_label)

    # Non-Vectorized VERSION - Runs forever.....
    # pred_y = np.empty(0)
    # # Iterate through all test observations and find closest neighbors -> get the most common label
    # for observation in test_x:
    #     observation_distances = np.empty(0)
    #     # 1. Get distances for all training set neighbors
    #     for neighbor in train_x:
    #         currentDistance = distance.euclidean(observation, neighbor)
    #         # print(f"Observation: {observation}")
    #         # print(f"neighbor: {neighbor}")
    #         # print(f"Distance: {currentDistance}")
    #         # print("----------------------------------")
    #         observation_distances = np.append(
    #             observation_distances, currentDistance)
    #     # 2. select the num_nn closest (i.e. shortest) distances
    #     closest_neighbors = observation_distances.argsort()[:num_nn]
    #     print("----------------------------------")
    #     print("Closest Neighbors")
    #     # Print the indices, distances, and flabels of the closest neighbors
    #     for neighbor in closest_neighbors:
    #         print(f"Index: {neighbor} | Distance: {observation_distances[neighbor]}  Label: {train_y[neighbor]}")
    #     # 3. Pick the most common label of the num_nn nearest neighbors
    #     closest_labels = [train_y[i] for i in closest_neighbors]
    #     print(f"Closest Labels: {closest_labels}")
    #     # Select the mode or argmax
    #     predicted_label = stats.mode(closest_labels)[0][0]
    #     print(f"Predicted Label: {predicted_label}")
    #     # Set the predicted label for the test observation
    #     pred_y = np.append(pred_y, predicted_label)

    return pred_y


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

def plot_kNN_accuracy(k_values, kNN_accuracies, sample_size):
    # Plot settings
    plt.grid()
    plt.figure(1)
    plt.title(f"kNN Evaluation (n={sample_size})")
    plt.xlabel("k Size")
    plt.ylabel("Accuracy (%)")
    plt.ylim(round(min(kNN_accuracies)), round(max(kNN_accuracies)+1))
    plt.yticks(np.arange(round(min(kNN_accuracies)-1), round(max(kNN_accuracies)+1)))
    plt.xlim(0,10)
    plt.xticks(np.arange(1, 11, step=2))
    plt.plot(k_values, kNN_accuracies, '-o' ,label=f"{sample_size}")
    plt.savefig(f"./kNN_plots/kNN_n{sample_size}_evaluation.png", bbox_inches='tight')
    ax = plt.gca()
    ax.lines.pop(0)
    plt.figure(2)
    plt.plot(k_values, kNN_accuracies, '-o' ,label=f"{sample_size}")
    return plt

def get_sub_sample_indices(array, size):
    """Sub Sample an array by providing the number of samples needed

    Args:
        array (np.array): The array to sample, we just need its length
        size (int): The number of indices from the array to generate

    Returns:
        np.array: An array containing randomly sampled indices
    """

    sub_sampled_indices = np.random.choice(np.arange(len(array)), size, replace=False)
    return sub_sampled_indices


def main():
    datasetPath = './letter-recognition.data'
    listClasses = []
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

    # Run kNN and Pocket on these sample sizes
    sub_sample_sizes = [100, 1000, 2000, 5000, 10000, 15000]
    
    # kNN Section -------------------------------------------------------------
    k_values = [1, 3, 5, 7, 9]
    kNN_start_time = datetime.datetime.now() 
    for sample_size in sub_sample_sizes:
        # Sub sample the training data
        sample_indices = get_sub_sample_indices(trainX, sample_size)
        trainXSample = trainX[sample_indices]
        trainYSample = trainY[sample_indices]
        # Initialize this sub samples kNN accuracy array
        kNN_accuracies = []
        kNN_run_times = []
        # Run kNN for each required k value
        for k_value in k_values:
            time_before = datetime.datetime.now()
            print(f"Starting kNN (k = {k_value}, n = {sample_size})")
            pred_y = test_knn(trainXSample, trainYSample, testX, k_value)
            time_after = datetime.datetime.now()
            run_time = time_after - time_before
            kNN_run_times.append(run_time)
            kNN_acc = compute_accuracy(testY, pred_y)
            kNN_accuracies.append(kNN_acc)
        #print(f"Run Time: {kNN_run_times}")   
        #print(f"Accuracy : {kNN_accuracies}%")
        # Add to plot
        plt = plot_kNN_accuracy(k_values, kNN_accuracies, sample_size)
    kNN_end_time = datetime.datetime.now()
    kNN_total_run_time = kNN_end_time - kNN_start_time
    print(f"Total Run Time: {kNN_total_run_time}")
    plt.figure(2)
    plt.title(f"kNN Evaluation")
    plt.xlabel("k Size")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xlim(0,10)
    plt.xticks(np.arange(1, 11, step=2))
    plt.yticks(np.arange(0, 110, step=10))
    plt.legend(loc=(1.05, 0.5), title='Sample Size (n)')
    plt.savefig("kNN_evaluation.png", bbox_inches='tight')



if __name__ == "__main__":
    main()
