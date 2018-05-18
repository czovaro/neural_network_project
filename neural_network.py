import sys
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def load_data(data_name):
    '''This method receives a csv file name.
       Returns two lists containg the input values and the output values from the file.'''
    input_dataset = []
    output_dataset = []
    with open(data_name) as _file:
        data = csv.reader(_file, delimiter=',')
        for line in data:
            line = [int(item) for item in line]
            input_dataset.append(line[0:64])
            output_dataset.append(line[64])
    return input_dataset, output_dataset


def score(predicted, test_output):
    '''This method receives two arrays, the first one with the predicted values and the second one with the
    real values. Returns the accuracy.'''
    hits = 0
    for i in range(len(test_output)):
        if predicted[i] == test_output[i]:
            hits += 1
    return (hits / len(predicted))


def main():
    train_input, train_output = load_data(sys.argv[1])
    classifier = MLPClassifier(hidden_layer_sizes=(9, 9),
                               learning_rate_init=0.001,
                               max_iter=3000,
                               tol=0.000001,
                               verbose=True)
    classifier.fit(train_input, train_output)
    print("Activation Function: ", classifier.out_activation_)
    test_input, test_output = load_data(sys.argv[2])
    predicted = classifier.predict(test_input)
    accuracy = score(predicted, test_output)
    matrix = confusion_matrix(test_output, predicted)
    print("Accuracy", round(accuracy*100, 1), "%")
    print("Confusion Matrix: \n", matrix)


if __name__ == "__main__":
    main()
