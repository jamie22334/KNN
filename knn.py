import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.feature_list = list()
        self.label_dict = dict()

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.feature_list = features

        for i in range(len(features)):
            self.label_dict[tuple(features[i])] = int(labels[i])

        # print("features: " + str(self.feature_list))
        # print("labels: " + str(self.label_dict))

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        test_cases = features
        predicted_labels = list()

        for case in test_cases:
            k_neighbors = self.get_k_neighbors(case)
            majority_label = self.get_majority_label(k_neighbors)
            predicted_labels.append(majority_label)

        print("predicted_labels: " + str(predicted_labels))
        return predicted_labels

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        distance_list = list()
        k_labels = list()
        # sorted_distance = list()

        for f in self.feature_list:
            distance_list.append(self.distance_function(f, point))

        out_arr = np.argsort(distance_list)
        # print("argsort: " + str(out_arr))

        for i in range(self.k):
            if i < len(self.feature_list):
                feature_tuple = tuple(self.feature_list[out_arr[i]])
                k_labels.append(self.label_dict[feature_tuple])
                # sorted_distance.append(distance_list[out_arr[i]])

        # print("sorted distance: " + str(sorted_distance))

        if self.k != len(k_labels):
            print("k different:\n")
            print("k: " + str(self.k) + ", k_neighbors length: " + str(len(k_labels)))
        return k_labels

    def get_majority_label(self, k_labels):
        counter_dict = dict()

        for label in k_labels:
            if label not in counter_dict:
                counter_dict[label] = 0
            counter_dict[label] = counter_dict[label] + 1
        # print("counter map: " + str(counter_dict))

        max_count = -1
        max_label = -1
        for key in counter_dict:
            if counter_dict[key] > max_count:
                max_count = counter_dict[key]
                max_label = key
        # print("max label: " + str(max_label))
        return max_label


if __name__ == '__main__':
    print(np.__version__)
