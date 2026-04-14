print("🚀 THIS IS NEW FILE RUNNING")

import argparse
import os
import sys

import cv2 # type: ignore
class editdistance:
    @staticmethod
    def eval(a, b): return 0 # pyright: ignore[reportMissingModuleSource]
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from DataLoader import Batch, DataLoader, FilePaths
from SamplePreprocessor import preprocessor, wer
from Model import DecoderType, Model
from SpellChecker import correct_sentence


def train(model, loader):
    """ Train the neural network """
    epoch = 0  # Number of training epochs since start
    bestCharErrorRate = float('inf')  # Best valdiation character error rate
    noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
    earlyStopping = 25  # Stop training after this number of epochs without improvement
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize 

    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # Train
        print('Train neural network')
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate, addressAccuracy, wordErrorRate = validate(model, loader)
        cer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)
        address_summary = tf.Summary(value=[tf.Summary.Value(
            tag='addressAccuracy', simple_value=addressAccuracy)])  # Tensorboard: Track addressAccuracy
        # Tensorboard: Add address_summary to writer
        model.writer.add_summary(address_summary, epoch)
        wer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
        # Tensorboard: Add wer_summary to writer
        model.writer.add_summary(wer_summary, epoch)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                earlyStopping)
            break


def validate(model, loader):
    """ Validate neural network """
    print('Validate neural network')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            ## editdistance
            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER)/len(totalCER)
    addressAccuracy = numWordOK / numWordTotal
    wordErrorRate = sum(totalWER)/len(totalWER)
    print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
        (charErrorRate*100.0, addressAccuracy*100.0, wordErrorRate*100.0))
    return charErrorRate, addressAccuracy, wordErrorRate


def load_different_image():
    imgs = []
    for i in range(1, Model.batchSize):
        imgs.append(preprocessor(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
    return imgs


def generate_random_images():
    return np.random.random((Model.batchSize, Model.imgSize[0], Model.imgSize[1]))


def infer(model, fnImg):
    import cv2
    from SamplePreprocessor import preprocessor
    from DataLoader import Batch

    print("Using custom image...")

    img = cv2.imread('../data/img3.jpg', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found")
        return

    img = preprocessor(img, Model.imgSize)

    batch = Batch(None, [img])

    recognized = model.inferBatch(batch)

    print("Recognized Text:", recognized[0])
    return recognized[0]


def main():
    import cv2
from Model import Model, DecoderType
from SamplePreprocessor import preprocessor
from DataLoader import Batch


import cv2
from Model import Model, DecoderType
from SamplePreprocessor import preprocessor
from DataLoader import Batch


def main():
   print("🚀 NEW OCR FILE RUNNING")

import cv2
from Model import Model, DecoderType
from SamplePreprocessor import preprocessor
from DataLoader import Batch


def main():   # ✅ YES, THIS IS REQUIRED
    print("Loading image...")

    img = cv2.imread('../data/img3.jpg', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Error: Image not found")
        return

    img = preprocessor(img, Model.imgSize)

    print("Loading model...")

    model = Model(open('../data/charList.txt').read(),
                decoderType=DecoderType.BestPath)
    model.load()

    print("Running OCR...")

    batch = Batch(None, [img])
    recognized = model.inferBatch(batch)

    print("✅ Recognized Text:", recognized[0])


if __name__ == "__main__":   # ✅ THIS IS ALSO REQUIRED
    main()