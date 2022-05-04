function [documents] = preProcessReviews(textData)
%Converts the text to lower case
cleanTextData = lower(textData);
%Tokenizes the text
documents = tokenizedDocument(cleanTextData);
%Removes punctuation
documents = erasePunctuation(documents);
%Removes a list of stop words created by MATLAB
documents = removeStopWords(documents);
end