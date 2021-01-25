# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from BaggingWithTrees import *
import numpy as np


def main():
    dataSet = pd.read_csv('titanikData.csv')
    dataSetTest = pd.read_csv('titanikTest.csv', names=['pclass', 'age', 'gender', 'survived'])
    dataSet = CategoricalToNumerical(dataSet)
    dataSetTest = CategoricalToNumerical(dataSetTest)
    finalDataSetTest = dataSetTest.drop('survived', axis=1)
    uniqueDataFrame = dataSet.drop_duplicates()
    predictions = list()
    for i in range(100):
        # Create sub sample and convert it to DataFrame.
        sample = createSubSamples(uniqueDataFrame)
        prediction = createTree(sample, dataSetTest, finalDataSetTest)
        predictions.append(prediction)
    array = np.array(predictions)
    results = [sum(x) for x in zip(*array)]
    for i in range(len(results)):
        if results[i] > 50:
            results[i] = 1
        else:
            results[i] = 0
    decisionsDF = pd.DataFrame(results, columns=["predict"])
    final = pd.concat([dataSetTest, decisionsDF], axis=1)
    print(final)
    print("Accuracy - " + str(accuracy_score(final.survived, final.predict)))


if __name__ == '__main__':
    main()

