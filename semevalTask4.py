import pickle
import string
import XMLParser
import nltk
import scipy
from nltk.corpus import conll2000
from nltk.chunk.util import conlltags2tree


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): # [_code-unigram-chunker-constructor]
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) # [_code-unigram-chunker-buildit]

    def parse(self, sentence): # [_code-unigram-chunker-parse]
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        #print "input to conlltags", conlltags
        return conlltags2tree(conlltags)


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using maximum entropy; should also try NB
    """

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            #print "tagged", tagged_sent
            untagged_sent = nltk.tag.untag(tagged_sent)
            #print "untagged", untagged_sent
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = chunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        #megam doesn't work in below - seems hard to get working; cg also doesn't work
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='iis',
            trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = chunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                           sent]
                        for sent in train_sents]
        #print tagged_sents
        self.tagger = ConsecutiveChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        #print "parse result:", tagged_sents
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        #print "conlltags:", conlltags
        return conlltags2tree(conlltags)

    def evaluate(self, gold):
        chunkscore = nltk.ChunkScore()
        for tagged_sent in gold:
            #print "true:", tagged_sent
            #score thinks things should be in trees
            chunkscore.score(conlltags2tree(tagged_sent), self.parse([(w,t) for (w,t,_c) in tagged_sent]))
        return chunkscore


def chunk_features(sentence, i, history):
    """ Get features for sentence at position i with history being tags seen so far.
    Returns: dictionary of features.
    """
    word, pos = sentence[i]
    if i == 0:
        prevw, prevpos = "<START>", "<START>"
        prevtag = "<START>"
    else:
        prevw, prevpos = sentence[i-1]
        prevtag = history[i-1]
    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
    else:
        nextw, nextpos = sentence[i+1]
    return {'word': word, 'pos': pos, 'prevpos': prevpos, 'nextpos': nextpos, 'prevtag': prevtag}


def train_and_test(filename):
    traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = traind['iob'][:split_size]
    test = traind['iob'][split_size:]
    chunker = ConsecutiveChunker(train)
    #the following line does not work, probably need to represent it differently
    print chunker.evaluate(test)


#not yet in use
def create_features(token, tag):
    if token[0] == '#':
        yield "hashtag"
        t = token[1:]
        yield "token={}".format(t.lower())
        yield "token,tag={},{}".format(t, tag)
    elif token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,tag={},{}".format(token, tag)


if __name__ == '__main__':
    pass
