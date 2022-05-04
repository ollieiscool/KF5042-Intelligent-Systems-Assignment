%Load embeddings
rng('default');
emb = fastTextWordEmbedding;

%Load dataset
reviews = readtable("train.csv", "TextType", "string");
humanScore = reviews.user_suggestion;

%Partition the dataset into training and testing sets
reviews.user_suggestion = categorical(reviews.user_suggestion);
cvp = cvpartition(humanScore, 'HoldOut', 0.2);
trainSet = reviews(training(cvp),:);
testSet = reviews(test(cvp),:);

%Pre-process reviews
reviewsTrain = preProcessReviews(trainSet.user_review);
reviewsTest = preProcessReviews(testSet.user_review);
trainY = trainSet.user_suggestion;
testY = testSet.user_suggestion;

%Encode the words to numerical indices
enc = wordEncoding(reviewsTrain);
seqLength = 300;
trainX = doc2sequence(enc, reviewsTrain, "Length", seqLength);
testX = doc2sequence(enc, reviewsTest, "Length", seqLength);

%LSTM training
embDimension = 50;
numHiddenUnits = 80;
numWords = enc.NumWords;
layers = [...
    sequenceInputLayer(1)
    wordEmbeddingLayer(embDimension, numWords)
    lstmLayer(numHiddenUnits, "OutputMode", "last")
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions("adam", ...
    "MiniBatchSize", 16, ...
    "GradientThreshold", 2, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {testX, testY}, ...
    "Plots", "training-progress", ...
    "Verbose", false);
net = trainNetwork(trainX, trainY, layers, options);
predictY = classify(net, testX);
plotconfusion(testY, predictY);