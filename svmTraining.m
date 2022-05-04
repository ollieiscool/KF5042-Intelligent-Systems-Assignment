rng('default');
emb = fastTextWordEmbedding;

words = [wordsPos;wordsNeg];
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPos)) = "Positive";
labels(numel(wordsPos) + 1 : end) = "Negative";
labWords = table(words, labels, 'VariableNames', {'Word', 'Label'});
duffWords = ~isVocabularyWord(emb, labWords.Word); labWords(duffWords, :) = [];

trainX = word2vec(emb, labWords.Word);
trainY = labWords.Label;
trainedSVM = fitcsvm(trainX, trainY);