%Read words
fidPositive = fopen(fullfile('opinion-lexicon-English', 'positive-words.txt'));
fidNegative = fopen(fullfile('opinion-lexicon-English', 'negative-words.txt'));
%Skip reading comments in the text file
C = textscan(fidPositive, '%s', 'CommentStyle', ';');
D = textscan(fidNegative, '%s', 'CommentStyle', ';');
%Convert cell arrays to string
wordsPos = string(C{1});
wordsNeg = string(D{1});
%Close file
fclose all;

%Define hashtable
lexHash = java.util.Hashtable;
[posSize, ~] = size(wordsPos);
[negSize, ~] = size(wordsNeg);
for ii = 1:posSize
    lexHash.put(wordsPos(ii,1),1);
end
for ii = 1:negSize
    lexHash.put(wordsNeg(ii,1),-1);
end