import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(real_labels)):
        if predicted_labels[i] == 1 and real_labels[i] == 1:
            tp = tp + 1
        elif predicted_labels[i] == 1 and real_labels[i] == 0:
            fp = fp + 1
        elif predicted_labels[i] == 0 and real_labels[i] == 1:
            fn = fn + 1

    # print("real: " + str(real_labels))
    # print("predict: " + str(predicted_labels))
    # print("tp: " + str(tp) + ", fp: " + str(fp) + ", fn: " + str(fn))
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall != 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum_distance = sum(pow(abs(a-b), 3) for a, b in zip(point1, point2))

        return pow(sum_distance, 1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.sqrt(sum(pow(a - b, 2) for a, b in zip(point1, point2)))

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.inner(point1, point2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if np.sqrt(np.dot(point1, point1)) * np.sqrt(np.dot(point2, point2)) != 0:
            similarity = np.dot(point1, point2) / (np.sqrt(np.dot(point1, point1)) * np.sqrt(np.dot(point2, point2)))
        else:
            similarity = 1

        return 1 - similarity

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        arr1 = np.array(point1)
        arr2 = np.array(point2)
        square_sum = np.inner(arr1 - arr2, arr1 - arr2)

        return -np.exp(-0.5 * square_sum)


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        self.distance_function_map = dict()
        self.scaler_map = dict()

        self.distance_function_map['euclidean'] = 5
        self.distance_function_map['minkowski'] = 4
        self.distance_function_map['gaussian'] = 3
        self.distance_function_map['inner_prod'] = 2
        self.distance_function_map['cosine_dist'] = 1
        print(self.distance_function_map)

        self.scaler_map['min_max_scale'] = 2
        self.scaler_map['normalize'] = 1

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        max_f1 = -1
        for func_key in distance_funcs:
            for k in range(1, 30, 2):
                model = KNN(k, distance_funcs[func_key])
                model.train(x_train, y_train)
                predicted_y = model.predict(x_val)
                current_f1 = f1_score(y_val, predicted_y)
                # handle tie!!!
                if current_f1 > max_f1:
                    # print("replace")
                    # print("current func key: " + str(func_key))
                    # print("best func key: " + str(self.best_distance_function))
                    max_f1 = current_f1
                    self.best_k = k
                    self.best_distance_function = func_key
                    self.best_model = model
                elif current_f1 == max_f1:
                    # print("current func key: " + str(func_key))
                    # print("best func key: " + str(self.best_distance_function))
                    if self.distance_function_map[func_key] > self.distance_function_map[self.best_distance_function]:
                        self.best_k = k
                        self.best_distance_function = func_key
                        self.best_model = model
                    elif self.distance_function_map[func_key] == self.distance_function_map[self.best_distance_function]:
                        self.best_k = min(self.best_k, k)
                        self.best_model = KNN(self.best_k, distance_funcs[func_key])

        # print("best k: " + str(self.best_k))
        # print("best distance function: " + str(self.best_distance_function))

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        max_f1 = -1
        for scalar_key in scaling_classes:
            scalar = scaling_classes[scalar_key]()
            x_train_scaled = scalar(x_train)
            x_val_scaled = scalar(x_val)

            for func_key in distance_funcs:
                for k in range(1, 31, 2):
                    model = KNN(k, distance_funcs[func_key])
                    model.train(x_train_scaled, y_train)
                    predicted_y = model.predict(x_val_scaled)
                    current_f1 = f1_score(y_val, predicted_y)
                    # handle tie!!!
                    if current_f1 > max_f1:
                        max_f1 = current_f1
                        self.best_k = k
                        self.best_distance_function = func_key
                        self.best_scaler = scalar_key
                        self.best_model = model
                    elif current_f1 == max_f1:
                        if self.scaler_map[scalar_key] > self.scaler_map[self.best_scaler]:
                            self.best_k = k
                            self.best_distance_function = func_key
                            self.best_scaler = scalar_key
                            self.best_model = model
                        elif self.scaler_map[scalar_key] == self.scaler_map[self.best_scaler]:
                            if self.distance_function_map[func_key] > self.distance_function_map[self.best_distance_function]:
                                self.best_k = k
                                self.best_distance_function = func_key
                                self.best_model = model
                            elif self.distance_function_map[func_key] == self.distance_function_map[self.best_distance_function]:
                                self.best_k = min(self.best_k, k)
                                self.best_model = KNN(self.best_k, distance_funcs[func_key])


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_list = list()
        # features = [[3, 4], [1, -1], [0, 0]]

        for point in features:
            all_zero = True
            for f in point:
                if f != 0:
                    all_zero = False

            if all_zero:
                normalized_list.append(point)
            else:
                arr = np.array(point)
                dist = np.sqrt(np.inner(arr, arr))
                normalized_list.append(list(arr / dist))

        # print(normalized_list)
        return normalized_list


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_time = True
        self.min_list = list()
        self.max_list = list()

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # features = [[2, -1], [-1, 5], [0, 0]]
        min_max_result = list()
        if self.first_time:
            self.first_time = False
            self.min_list = np.full(len(features[0]), np.inf)
            self.max_list = np.full(len(features[0]), -np.inf)

            for point in features:
                for i in range(len(point)):
                    self.min_list[i] = min(self.min_list[i], point[i])
                    self.max_list[i] = max(self.max_list[i], point[i])

        for point in features:
            arr = np.array(point)
            subList = list()
            for i in range(len(point)):
                if self.max_list[i] - self.min_list[i] != 0:
                    subList.append((arr[i] - self.min_list[i]) / (self.max_list[i] - self.min_list[i]))
                else:
                    subList.append(0)
            min_max_result.append(subList)

        # print(min_max_result)
        return min_max_result
