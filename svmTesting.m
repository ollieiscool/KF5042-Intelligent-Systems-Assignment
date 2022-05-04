%Load the dataset and pre-process text
reviews = readtable("train.csv", "TextType", "string");
textData = reviews.user_review;
humanScore = reviews.user_suggestion;
processedRevs = preProcessReviews(textData);

%Remove words not in the embeddings
duffWords = ~isVocabularyWord(emb, processedRevs.Vocabulary);
removeWords(processedRevs, duffWords);

%Convert words to vectors and predict the sentiment of the reviews using
%the SVM
sentimentScore = zeros(size(processedRevs));
for ii = 1 : processedRevs.length
    vocabWords = processedRevs(ii).Vocabulary;
    vectors = word2vec(emb, vocabWords);
    [~, predScores] = predict(trainedSVM, vectors);
    %If NaN values are found use the table of words without them if not 
    %use the original table of words
    noMissing = rmmissing(predScores);
    if (isempty(noMissing) == true)
        sentimentScore(ii) = mean(predScores(:, 1));
    else
        sentimentScore(ii) = mean(noMissing(:, 1));
    end
    
    if (isnan(sentimentScore(ii)) == true)
        sentimentScore(ii) = 0;
    end

    %Find the coverage, tp, tn, fp, fn, accuracy, precision, recall and F1
    %score.
fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(processedRevs(ii)), sentimentScore(ii), humanScore(ii));
end

numZeros = sum(sentimentScore == 0);
numCovered = numel(sentimentScore) - numZeros;
fprintf("Total positives and negatives (coverage): %2.2f%%, Number: %d, Not found or neutral: %d\n", (numCovered*100)/numel(sentimentScore), numCovered, numZeros);
truePos = sentimentScore((sentimentScore > 0) & (humanScore == 1));
falsePos = sentimentScore((sentimentScore > 0) & (humanScore ~= 1));
trueNeg = sentimentScore((sentimentScore < 0) & (humanScore == 0));
falseNeg = sentimentScore((sentimentScore < 0) & (humanScore ~= 0));
acc = (numel(truePos) + numel(trueNeg))*100/numCovered;
prec = numel(truePos)*100/(numel(truePos) + numel(falsePos));
rec = numel(truePos)*100/(numel(truePos) + numel(falseNeg));
f1Score = (2*prec*rec)/(prec + rec);
fprintf("Accuracy: %2.2f%%, TP: %d, FP: %d, TN: %d, FN: %d\n", acc, numel(truePos), numel(falsePos), numel(trueNeg), numel(falseNeg));
fprintf("Precision: %2.2f%%, Recall: %2.2f%%, F1 Score: %2.2f%%\n", prec, rec, f1Score);

%Add labels to each of the reviews for their human-set score and predicted
%score
predLabels = zeros(size(sentimentScore));
humanLabels = zeros(size(sentimentScore));
for ii = 1 : processedRevs.length
    if humanScore(ii) == 1
        humanLabels(ii) = 1;
    elseif humanScore(ii) == 0
        humanLabels(ii) = 0;
    end
    if sentimentScore(ii) > 0
        predLabels(ii) = 1;
    elseif sentimentScore(ii) < 0
        predLabels(ii) = 0;
    end
end
%Plot confusion
humanLabels = categorical(humanLabels);
predLabels = categorical(predLabels);
plotconfusion(humanLabels, predLabels);