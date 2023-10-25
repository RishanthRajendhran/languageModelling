import argparse
import logging
import numpy as np
from pathlib import Path
from os.path import exists
import os
import glob
import torch
import pandas as pd
import transformers
import math
from tqdm import tqdm
import pickle as pkl

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-trainDir",
    help="Path to directory containing train files",
    default="./data/train"
)

parser.add_argument(
    "-valDir",
    help="Path to directory containing validation files",
    default="./data/dev"
)

parser.add_argument(
    "-testDir",
    help="Path to directory containing test files",
    default="./data/test"
)

parser.add_argument(
    "-vocabFile",
    help="Path to file containing vocabulary",
    default="./data/vocab.pkl"
)

# parser.add_argument(
#     "-out",
#     help="Path to directory where learned embedding should be saved",
#     default="./embeddings/"
# )

parser.add_argument(
    "-ngram",
    type=int,
    help="Size of ngram to consider",
    default=4
)

parser.add_argument(
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=10
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=64
)

parser.add_argument(
    "-learningRate",
    type=float,
    nargs="+",
    help="Learning rate(s) for optimizer",
    default=[0.01, 0.01, 0.0001]
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=-1
)
#---------------------------------------------------------------------------
def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from. 

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list
#---------------------------------------------------------------------------
def convert_line2idx(line, vocab):
    """ Converts a string into a list of character indices
    Input
    ------------
    line: str. A line worth of data
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    -------------
    line_data: List[int]. List of indices corresponding to the characters
                in the input line.
    """
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data
#---------------------------------------------------------------------------
def convert_files2idx(files, vocab):
    """ This method iterates over files. In each file, it iterates over
    every line. Every line is then split into characters and the characters are 
    converted to their respective unique indices based on the vocab mapping. All
    converted lines are added to a central list containing the mapped data.
    Input
    --------------
    files: List[str]. List of files in a particular split
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    ---------------
    data: List[List[int]]. List of lists where each inner list is a list of character
            indices corresponding to a line in the training split.
    """
    data = []

    for file in files:
        with open(file) as f:
            lines = f.readlines()
        
        for line in lines:
            toks = convert_line2idx(line, vocab)
            data.append(toks)

    return data
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")   
    return path
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#---------------------------------------------------------------------------
class TraditionalLM(torch.nn.Module):
    def __init__(self, vocab):
        super(TraditionalLM, self).__init__()
        self.vocab = vocab
        self.counts = {}
    
    #Batched input only
    #Does not return anything
    def train(self, inputs):
        for input in tqdm(inputs, desc="Train data"):
            if str(input[:-1]) not in self.counts.keys():
                self.counts[str(input[:-1])] = {}
            if str(input[-1]) not in self.counts[str(input[:-1])].keys():
                self.counts[str(input[:-1])][str(input[-1])] = 0
            self.counts[str(input[:-1])][str(input[-1])] += 1
        numParams = len([len(v.keys()) for v in self.counts.values()])
        logging.info("No. of parameters: {}".format(numParams))

    #Unbatched input only
    #Returns perplexity of input sentence
    def test(self, input):
        if len(input) == 0:
            return 0
        loss = 0
        for d in input:
            num, den = 1, len(self.vocab)
            if str(d[:-1]) in self.counts.keys():
                den += sum(self.counts[str(d[:-1])].values())
                if str(d[-1]) in self.counts[str(d[:-1])].keys():
                    num += self.counts[str(d[:-1])][str(d[-1])]
            loss -= np.log2(num/den)
        loss = loss*(1/len(input))
        perplexity = 2**loss
        return perplexity
#---------------------------------------------------------------------------
def getWindow(data, end, ngramSize, vocab, padToken="[PAD]", errTrace="getWindow"):
    if padToken not in vocab.keys():
        raise RuntimeError("[{}] Could not find {} in vocab!".format(errTrace, padToken))
    window = [vocab[padToken]]*ngramSize
    start = max(end-ngramSize, 0)
    window[-(end-start):] = data[start:end]
    return window
#---------------------------------------------------------------------------
def processData(data, ngramSize, vocab, test=False):
    proData = []
    for line in data:
        lineData = []
        for end in range(1, len(line)+1):
            lineData.append(getWindow(line, end, ngramSize, vocab))
        if not test:
            proData.extend(lineData)
        else: 
            proData.append(lineData)
    return proData
#---------------------------------------------------------------------------
def get_word2ix(path, errTrace="get_word2ix"):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            word2ix =  pkl.load(f)
    else: 
        raise ValueError("[{}] {} has invalid file extension!".format(errTrace, path))
    if "[PAD]" not in word2ix.keys():
        word2ix["[PAD]"] = max(word2ix.values())+1
    return word2ix
#---------------------------------------------------------------------------
def main(errTrace="main"):
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if args.ngram <= 0:
        raise ValueError("[{}] ngram has to be a positive number!".format(errTrace))
    
    args.trainDir = checkIfExists(args.trainDir, isDir=True, createIfNotExists=False)
    args.valDir = checkIfExists(args.valDir, isDir=True, createIfNotExists=False)
    args.testDir = checkIfExists(args.testDir, isDir=True, createIfNotExists=False)
    _ = checkIfExists(args.vocabFile, isDir=False, createIfNotExists=False)
    checkFile(args.vocabFile, fileExtension=".pkl")

    word2ix = get_word2ix(path=args.vocabFile)
    trainFiles = get_files(args.trainDir)
    valFiles = get_files(args.valDir)
    testFiles = get_files(args.testDir) 

    trainData = convert_files2idx(trainFiles, word2ix)
    valData = convert_files2idx(valFiles, word2ix)
    testData = convert_files2idx(testFiles, word2ix)

    proTrainData = processData(trainData, args.ngram, word2ix)
    proValData = processData(valData, args.ngram, word2ix)
    proTestData = processData(testData, args.ngram, word2ix, test=True)

    model = TraditionalLM(word2ix)
    logging.info("Training...")
    model.train(proTrainData)
    logging.info("Testing...")
    avgPerplexity = 0
    for d in tqdm(proTestData, desc="Test data"):
        avgPerplexity += model.test(d)
    # logging.info("Perplexity on test set: {:0.4f}".format(avgPerplexity))
    avgPerplexity /= len(proTestData)
    logging.info("Average Perplexity on test set: {:0.4f}".format(avgPerplexity))
#------------------- --------------------------------------------------------
if __name__=="__main__":
    main()