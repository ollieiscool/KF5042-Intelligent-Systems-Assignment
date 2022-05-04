%Load Embeddings
rng('default');
emb = fastTextWordEmbedding;

%Load words into a list and label them positive or negative
words = [wordsPos;wordsNeg];
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPos)) = "Positive";
labels(numel(wordsPos) + 1 : end) = "Negative";
labWords = table(words, labels, 'VariableNames', {'Word', 'Label'});
duffWords = ~isVocabularyWord(emb, labWords.Word); labWords(duffWords, :) = [];

%Train an SVM
trainX = word2vec(emb, labWords.Word);
trainY = labWords.Label;
trainedSVM = fitcsvm(trainX, trainY);