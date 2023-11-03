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

FORWARD_DIM = 200

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

parser.add_argument(
    "-out",
    help="Path to directory where outputs should be saved",
    default="./models/"
)

parser.add_argument(
    "-save",
    type=str,
    help="Path to directory where outputs are to be saved",
    default="./out/"
)

parser.add_argument(
    "-load",
    type=str,
    help="[Optional] Path to saved PyTorch model to load"
)

parser.add_argument(
    "-ngram",
    type=int,
    help="Sequence length",
    default=500
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
    default=5
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
    default=[0.0001, 0.00001, 0.000001]
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

parser.add_argument(
    "-embedDim",
    type=int, 
    help="Size of embedding dimension",
    default=50
)

parser.add_argument(
    "-hiddenSize",
    type=int, 
    help="Size of hidden dimension for LSTM block",
    default=200
)

parser.add_argument(
    "-numLayers",
    type=int, 
    help="Number of layers in LSTM block",
    default=1
)

parser.add_argument(
    "-biLSTM",
    action="store_true",
    help="Boolean flag to enable bi-directionality in LSTM block",
)

parser.add_argument(
    "-optimizer",
    choices=["adam", "adagrad"],
    help="Choice of optimizer to use for training",
    default="adagrad"
)

parser.add_argument(
    "-amsgrad",
    action="store_true",
    help="Boolean flag to enable amsgrad in optimizer"
)

parser.add_argument(
    "-lrScheduler",
    action="store_true",
    help="Boolean flag to employ learning rate scheduler during training"
)

parser.add_argument(
    "-testOnly",
    action="store_true",
    help="Boolean flag to only perform testing; Model load path must be provided when this is set to true"
)

parser.add_argument(
    "-numSamples",
    type=int,
    help="No. of samples to generate while testing",
    default=200
)

parser.add_argument(
    "-sampling",
    choices=["topP", "topK"],
    help="Type of sampling to use while testing",
    default="topK"
)

parser.add_argument(
    "-topP",
    type=float,
    help="top_p for nucleus sampling while testing",
    default=0.98
)

parser.add_argument(
    "-topK",
    type=int,
    help="top_k for topK sampling while testing",
    default=5
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
            line = line.strip()
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
class LanguageModel(torch.nn.Module):
    def __init__(self, vocab, embedDim, hiddenSize, numLayers=1, biLSTM=False, device="cpu"):
        super(LanguageModel, self).__init__()
        self.vocab = vocab
        self.embedDim = embedDim
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.biLSTM = biLSTM
        self.wEmb = torch.nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedDim,
        )
        self.lstmBlock = torch.nn.LSTM(
            input_size=embedDim, 
            hidden_size=hiddenSize, 
            num_layers=numLayers, 
            bias=True, 
            batch_first=True, 
            bidirectional=biLSTM, 
        )

        self.finalBlock = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(1+biLSTM)*hiddenSize, 
                out_features=FORWARD_DIM, 
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=FORWARD_DIM, 
                out_features=len(vocab), 
                bias=True
            )
        )
        
        self.device = device
        self.to(device)
    
    def getModelParams(self):
        return {
            "vocab": self.vocab, 
            "embedDim": self.embedDim, 
            "hiddenSize": self.hiddenSize, 
            "numLayers": self.numLayers, 
            "biLSTM": self.biLSTM, 
            "device": self.device,
        }

    def forward(self, input, h_0=None, c_0=None):
        #Batched input only (N, L)
        assert input.dim() == 2
        input = input.to(device=self.device)
        emb = self.wEmb(input)

        if h_0==None:
            h_0 = torch.autograd.Variable(torch.zeros(self.numLayers, input.size()[0], self.hiddenSize)).to(device=self.device)
        if c_0==None:
            c_0 = torch.autograd.Variable(torch.zeros(self.numLayers, input.size()[0], self.hiddenSize)).to(device=self.device)

        lstmOutput, (h, c) = self.lstmBlock(emb, (h_0, c_0))

        output = self.finalBlock(lstmOutput.reshape(-1, lstmOutput.shape[-1])).reshape(lstmOutput.shape[0], lstmOutput.shape[1], -1)

        #D - 2 if biLSTM else 1
        #output - (N, L, vocabSize)
        #h - (D*numLayers, N, hiddenSize)
        #c - (D*numLayers, N, hiddenSize)
        return output, h, c

    def to(self, device):
        self.device = device 
        self = super(LanguageModel, self).to(device)
        return self 

    def numParams(self):
        return sum(param.numel() for param in self.parameters())
#---------------------------------------------------------------------------
class NGramDataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        input = torch.tensor(self.data[item])
        target = torch.tensor(self.target[item])
        return input, target
#---------------------------------------------------------------------------
def collateBatch(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels
#---------------------------------------------------------------------------
def createDataLoader(data, target, batchSize, shuffle=True):
    ds = NGramDataset(
        data = data, 
        target=target,
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=shuffle,
        collate_fn=collateBatch,
    )
#---------------------------------------------------------------------------
def getWindow(data, start, ngramSize, vocab, padToken="[PAD]", errTrace="getWindow"):
    if padToken not in vocab.keys():
        raise RuntimeError("[{}] Could not find {} in vocab!".format(errTrace, padToken))
    window = [vocab[padToken]]*ngramSize
    end = min(start+ngramSize, len(data))
    window[0:(end-start)] = data[start:end]
    return window
#---------------------------------------------------------------------------
def processData(data, ngramSize, vocab, errTrace="processData"):
    proData = []
    proTarget = []
    frequencyCounts = {}
    for v in vocab:
        frequencyCounts[v] = 0
    for line in data:
        vals, counts = np.unique(line, return_counts=True)
        for curInd, curVal in enumerate(vals):
            if curVal not in frequencyCounts.keys():
                curVal = "<unk>"
                if curVal not in frequencyCounts.keys():
                    raise RuntimeError("[{}] Could not find '{}' in vocabulary!".format(errTrace, curVal))
            frequencyCounts[curVal] += counts[curInd]
        for start in range(0, len(line), ngramSize):
            proData.append(getWindow(line, start, ngramSize, vocab))
            proTarget.append(getWindow(line, start+1, ngramSize, vocab))
    return proData, proTarget, frequencyCounts
#---------------------------------------------------------------------------
def get_word2ix(path, errTrace="get_word2ix"):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            word2ix =  pkl.load(f)
    else: 
        raise ValueError("[{}] {} has invalid file extension!".format(errTrace, path))
    if "<unk>" not in word2ix.keys():
        word2ix["<unk>"] = max(word2ix.values())+1
    if "[PAD]" not in word2ix.keys():
        word2ix["[PAD]"] = max(word2ix.values())+1
    return word2ix
#---------------------------------------------------------------------------
def trainModel(model, dataLoader, lossFunction, optimizer, vocab, device="cpu", scheduler=None, maxSteps=-1, logSteps=1000, dataDesc="Train data"):
    model.to(device)
    model.train()

    losses = []
    corrPreds = 0
    numExamples = 0
    numBatch = 0
    numSteps = 0
    for inputs, targets in tqdm(dataLoader, desc=dataDesc):
        numBatch += 1
        numExamples += inputs.numel()
        outputs, _, _ = model(inputs)
        targets = targets.to(device)
        
        _, preds = torch.max(outputs, dim=-1)

        loss = lossFunction(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

        if numSteps%logSteps == 0:
            logging.info(f"\nBatch: {numBatch}/{len(dataLoader)}, Loss: {loss.item()}")

        #We are not interested in [PAD] tokens
        preds[targets == vocab["[PAD]"]] = vocab["[PAD]"]

        corrPreds += torch.sum(preds.reshape(-1) == targets.reshape(-1))
        losses.append(loss.item())
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        #Backwardpropagate the losses
        loss.backward()
        # #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
        numSteps += 1
        if maxSteps and numSteps >= maxSteps:
            break
    if scheduler:
        scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def evalModel(model, lossFunction, dataLoader, vocab, device="cpu", dataDesc="Test batch"):
    model.eval()
    with torch.no_grad():
        losses = []
        corrPreds = 0
        numExamples = 0
        numBatch = 0
        numSteps = 0
        perplexity = 0
        numInstances = 0
        for inputs, targets in tqdm(dataLoader, desc=dataDesc):
            numBatch += 1
            numInstances += len(inputs)
            numExamples += inputs.numel()
            outputs, _, _ = model(inputs)
            targets = targets.to(device)
            _, preds = torch.max(outputs, dim=-1)

            loss = lossFunction(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)).reshape(inputs.shape[0], -1)
            nonPadMask = inputs != vocab["[PAD]"]

            # perplexity += (2**(loss.mean(dim=-1))).sum().item()
            perplexity += (np.exp(loss.cpu().sum(dim=-1)/nonPadMask.sum(dim=-1))).sum().item()

            #We are not interested in [PAD] tokens
            preds[targets == vocab["[PAD]"]] = vocab["[PAD]"]

            corrPreds += torch.sum(preds.reshape(-1) == targets.reshape(-1))
            losses.append(loss.mean().item())
            numSteps += 1
    return corrPreds.double()/numExamples, np.mean(losses), perplexity/numInstances
#---------------------------------------------------------------------------
def topKSampling(probs, top_k):
    probs, inds = probs.sort(descending=True)
    probsK, indsK = probs[:top_k], inds[:top_k]
    renormProbs = (probsK)/torch.sum(probsK)
    chosenTok = torch.multinomial(renormProbs, 1)
    return indsK[chosenTok]
#---------------------------------------------------------------------------
def nucleusSampling(probs, top_p):
    probs, inds = probs.sort(descending=True)
    cumProbs = torch.cumsum(probs, dim=0)
    thresh = [(i, c) for i, c in enumerate(cumProbs) if c>top_p][0][0]
    chosenTok = 0
    if thresh:
        renormProbs = (probs[:thresh])/torch.sum(probs[:thresh])
        chosenTok = torch.multinomial(renormProbs, 1)
    return inds[chosenTok]
#---------------------------------------------------------------------------
def testModel(model, data, numSamples, vocab, topSampling, samplingFn=topKSampling, device="cpu", dataDesc="Test data"):
    model.eval()
    with torch.no_grad():
        predictions = []
        for d in tqdm(data, desc=dataDesc):
            inputs = torch.tensor(d).reshape(1,-1)
            outputs, h, c = model(inputs)
            probs = torch.nn.Softmax(dim=-1)(outputs)
            for inp_i, prob_i, h_i, c_i in zip(inputs, probs, h.transpose(0, 1), c.transpose(0, 1)):
                prob_i = prob_i[-1]
                curPreds = inp_i.tolist()
                for _ in range(numSamples):
                    nextTok = samplingFn(prob_i.flatten(), topSampling)
                    curPreds.append(nextTok.item())
                    if nextTok.item() == vocab["[PAD]"]:
                        break
                    out_i, h_i, c_i = model(nextTok.reshape(1,1), h_i.unsqueeze(dim=1), c_i.unsqueeze(dim=1))
                    out_i, h_i, c_i = out_i.squeeze(dim=0), h_i.squeeze(dim=1), c_i.squeeze(dim=1)
                    prob_i = torch.nn.Softmax(dim=-1)(out_i)
                predictions.append(curPreds)
    return predictions
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
    
    if not args.testOnly:
        args.trainDir = checkIfExists(args.trainDir, isDir=True, createIfNotExists=False)
        args.valDir = checkIfExists(args.valDir, isDir=True, createIfNotExists=False)
    args.testDir = checkIfExists(args.testDir, isDir=True, createIfNotExists=False)

    _ = checkIfExists(args.vocabFile, isDir=False, createIfNotExists=False)
    checkFile(args.vocabFile, fileExtension=".pkl")

    args.out = checkIfExists(args.out, isDir=True, createIfNotExists=True)
    args.save = checkIfExists(args.save, isDir=True, createIfNotExists=True)

    word2ix = get_word2ix(path=args.vocabFile)
    if not args.testOnly:
        trainFiles = get_files(args.trainDir)
        valFiles = get_files(args.valDir)
    testFiles = get_files(args.testDir) 

    if not args.testOnly:
        trainData = convert_files2idx(trainFiles, word2ix)
        valData = convert_files2idx(valFiles, word2ix)
    testData = convert_files2idx(testFiles, word2ix)

    if not args.testOnly:
        proTrainData, proTrainTarget, frequencyCounts = processData(trainData, args.ngram, word2ix)
        proValData, proValTarget, _ = processData(valData, args.ngram, word2ix)
    proTestData, proTestTarget, _ = processData(testData, args.ngram, word2ix)

    if torch.cuda.is_available:
        device = "cuda"
    else: 
        device = "cpu"
    logging.info("Using device:{}".format(device))

    if args.load:
        model = torch.load(args.load)
        logging.info("Loaded LanguageModel from {}".format(args.load)) 
    else: 
        model = LanguageModel(word2ix, args.embedDim, args.hiddenSize, args.numLayers, args.biLSTM, device)
        logging.info("Created LanguageModel") 

    if not args.testOnly:
        trainDataLoader = createDataLoader(proTrainData, proTrainTarget, args.batchSize)
        valDataLoader = createDataLoader(proValData, proValTarget, args.batchSize)
    testDataLoader = createDataLoader(proTestData, proTestTarget, args.batchSize, shuffle=False)

    evalLossFunction = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=word2ix["[PAD]"],
    ).to(device)

    logging.info(args)

    if not args.testOnly:
        bestLearningRate = None
        bestValPerplexity = None
        bestValAcc = 0
        for expInd, learningRate in enumerate(args.learningRate):
            numTrainingSteps = args.numEpochs * len(trainDataLoader)
            maxSteps = args.maxSteps
            if maxSteps == -1:
                maxSteps = numTrainingSteps
            elif maxSteps > 0:
                maxSteps = math.ceil(maxSteps/len(trainDataLoader))
            else: 
                raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")
            
            if expInd:
                if args.load:
                    model = torch.load(args.load)
                    logging.info("Loaded LanguageModel from {}".format(args.load)) 
                else: 
                    model = LanguageModel(word2ix, args.embedDim, args.hiddenSize, args.numLayers, args.biLSTM, device)
                    logging.info("Created LanguageModel") 
            logging.info("Learning Rate: {}".format(learningRate))
            if args.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=learningRate, 
                    weight_decay=args.weightDecay,
                    amsgrad=args.amsgrad,
                )
            elif args.optimizer == "adagrad":
                optimizer = torch.optim.Adagrad(
                    model.parameters(), 
                    lr=learningRate, 
                    weight_decay=args.weightDecay,                
                )
            else:
                raise ValueError("[main] Invalid input to -optimizer: {}".format(args.optimizer))
            totalSteps = args.numEpochs
            if args.lrScheduler:
                scheduler = transformers.get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0.1*totalSteps,
                    # num_warmup_steps=2000,
                    # num_warmup_steps=0,
                    num_training_steps=totalSteps
                )
            else:
                scheduler = None
            countsSum = sum(list(frequencyCounts.values()))
            lossWeight = torch.zeros((len(word2ix,)))
            for v in word2ix.keys():
                lossWeight[word2ix[v]] = (1-(frequencyCounts[v]/countsSum))
            lossFunction = torch.nn.CrossEntropyLoss(
                weight=lossWeight,
                ignore_index=word2ix["[PAD]"],
            ).to(device)
            
            for epoch in range(args.numEpochs):
                curAcc, curLoss = trainModel(
                    model=model, 
                    dataLoader=trainDataLoader, 
                    lossFunction=lossFunction, 
                    optimizer=optimizer, 
                    vocab=word2ix,
                    device=device, 
                    scheduler=scheduler, 
                    maxSteps=maxSteps,
                )
                maxSteps -= len(trainDataLoader)
                valAcc, valLoss, valPerplexity = evalModel(
                    model, 
                    evalLossFunction,
                    valDataLoader, 
                    vocab=word2ix,
                    device=device,
                    dataDesc="Validation batch", 
                )

                logging.info("Epoch {}/{}\nTraining Loss: {:0.2f}\nTrain Accuracy: {:0.2f}%\nValidation Loss: {:0.2f}\nValidation Accuracy: {:0.2f}%\nValidation Perplexity: {:0.2f}".format(epoch+1, args.numEpochs, curLoss, curAcc*100, valLoss, valAcc*100, valPerplexity))
                logging.info("*****")

                if not bestValPerplexity or bestValPerplexity >= valPerplexity:
                    bestValPerplexity = valPerplexity
                    bestValAcc = valAcc
                    bestLearningRate = learningRate
                    torch.save(model, "{}model.pt".format(args.out))
                    logging.info("Model saved at '{}model.pt'".format(args.out))
                if maxSteps <= 0:
                    logging.info("Max steps reached!")
                    break
        logging.info("Best learning rate: {}".format(bestLearningRate))
        logging.info("Best model's validation perplexity: {}".format(bestValPerplexity))
        logging.info("Best model's validation accuracy: {:0.2f}%".format(bestValAcc*100))

        model = torch.load("{}model.pt".format(args.out))
        logging.info("No. of parameters in best model: {}".format(model.numParams()))
        testAcc, _, testPerplexity = evalModel(
            model, 
            evalLossFunction,
            testDataLoader, 
            vocab=word2ix,
            device=device,
            dataDesc="Test batch", 
        )
        logging.info("Test Accuracy: {:0.2f}%\nTest Perplexity: {:0.2f}".format(testAcc*100, testPerplexity))
    else:
        if args.sampling == "topK":
            predictions = testModel(model, testData, args.numSamples, word2ix, args.topK, topKSampling,device)
        elif args.sampling == "topP":
            predictions = testModel(model, testData, args.numSamples, word2ix, args.topP, nucleusSampling, device)
        else: 
            raise ValueError("[{}] Unrecognized sampling strategy: {}".format(errTrace, args.sampling))
        ix2word = {v:k for (k, v) in word2ix.items()}
        predictions = ["".join(list(map(lambda ind: ix2word[ind], curPreds))) for curPreds in predictions]
        with open("{}predictions.txt".format(args.save), "w") as f:
            for pred in predictions:
                f.write(pred)
                f.write("\n")
#------------------- --------------------------------------------------------
if __name__=="__main__":
    main()