import csv
import random
import math

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
    variance = sum([(x-mean(numbers))**2 for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def split(dataset, splitRatio):
    trainingSet = []
    trainingSetSize = int(len(dataset) * splitRatio)

    temp = list(dataset)
    while len(trainingSet) < trainingSetSize:
        trainingSet.append(temp.pop(random.randrange(len(temp))))
    return [trainingSet, temp]

def getMeanStDevByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    results = {}
    for genre, instances in separated.iteritems():
        temp = [(mean(i), stdev(i)) for i in zip(*instances)]
        del temp[-1]
        results[genre] = temp
    return results

def getProbabilitiesByClass(results, inputVector):
	probabilities = {}
	for genre, classSummaries in results.iteritems():
		probabilities[genre] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[genre] *= math.exp(-((x-mean)**2/(2*stdev**2))) * (1/(math.sqrt(2*math.pi) * stdev))
	return probabilities
			
def predict(results, inputVector):
    probabilities = getProbabilitiesByClass(results, inputVector)
    bestGenre = None
    topProbability = -1.0
    for genre, probability in probabilities.iteritems():
        if probability > topProbability:
            bestGenre = genre
            topProbability = probability
    return bestGenre

def getPredictions(results, testSet):
	predictions = []
	for i in range(len(testSet)):
		p = predict(results, testSet[i])
		predictions.append(p)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
    file = 'fma_metadata/mfcc_dataset.csv'
    
    lines = csv.reader(open(file, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    trainingSet, testSet = split(dataset, 0.75)
    print('Data set of size {0} split into:'.format(len(dataset)))
    print('Training Set: {0} rows'.format(len(trainingSet)))
    print('Test Set: {0} rows'.format(len(testSet)))

    accuracy = getAccuracy(testSet, getPredictions(getMeanStDevByClass(trainingSet), testSet))
    print('Accuracy: {0}%'.format(accuracy))

main()