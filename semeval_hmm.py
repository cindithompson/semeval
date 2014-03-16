import cPickle
import string
import XMLParser
import nltk
import scipy
from nltk.corpus import conll2000
from nltk.chunk.util import conlltags2tree
from nltk.tag import hmm
import re


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using HMM
    """

    def __init__(self, train_sents, sent_dict):
        train_set = []
        tag_set = []
        symbols = []
        for tagged_sent in train_sents:
            example = []
            for i, (wd_pos, tag) in enumerate(tagged_sent):
                tag_set.append(tag)
                pos = wd_pos[1]
                symbols.append(pos)
                #symbols.append(wd_pos)
                example.append((pos, tag))
            train_set.append(example)
        #print train_set
        trainer = hmm.HiddenMarkovModelTrainer(list(set(tag_set)), list(set(symbols)))
        #self.hmm = trainer.train_supervised(train_sents)
        self.hmm = trainer.train_supervised(train_set)

    def tag(self, sentence):
        #sentence is a list of (w,pos) tuples - elim the wds
        example = []
        for w,pos in sentence:
            example.append(pos)
        #return self.hmm.tag(sentence)
        iob_tags = self.hmm.tag(example)
        result = []
        for i in range(len(sentence)):
            result.append((sentence[i], iob_tags[i][1]))
        #print "tag res:",result
        return result


class ConsecutiveChunker(nltk.ChunkParserI):
    """
    Wrapper class for ConsecutiveChunkTagger. Note actual POS tagging is done
    before we even get here.
    """
    def __init__(self, train_sents, sent_dict):
        """Creates the tagger (actually called parse) via a ML approach
        within CCTagger
        """
        tagged_sents = [[((w, t), c) for (w, t, c) in
                           sent]
                        for sent in train_sents]
        #print tagged_sents
        self.tagger = ConsecutiveChunkTagger(tagged_sents, sent_dict)

    def parse(self, sentence):
        """ Return the "parse tree" version of an input POS-tagged sentence
         The leaves of the tree have tags for IOB labels
        """
        tagged_sents = self.tagger.tag(sentence)
        #print "in parse:",tagged_sents
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return conlltags2tree(conlltags)

    def evaluate(self, gold):
        """ nltk machinery computes Acc, P,R, and F-measure for trees.
        """
        chunkscore = nltk.ChunkScore()
        for tagged_sent in gold:
            print "true:", conlltags2tree(tagged_sent)
            #score thinks things should be in trees
            #print "guess:", self.parse([(w,t) for (w,t,_c) in tagged_sent])
            chunkscore.score(conlltags2tree(tagged_sent), self.parse([(w,t) for (w,t,_c) in tagged_sent]))
        return chunkscore


def chunk_features(sentence, i, sent_dict, history):
    """ Get features for sentence at position i with history being tags seen so far.
    Returns: dictionary of features.
    """
    word, pos = sentence[i]
    sentiment = sentiment_lookup(sent_dict, word, pos)
    if i == 0:
        prevw, prevpos = "<START>", "<START>"
        prevtag = "<START>"
        prev_sentiment = "<START>"
        #prev_negtive = "<START>"
    else:
        prevw, prevpos = sentence[i-1]
        prevtag = history[i-1]
        prev_sentiment = sentiment_lookup(sent_dict, prevw, prevpos)
    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
        next_sentiment = "<END>"
    else:
        nextw, nextpos = sentence[i+1]
        next_sentiment = sentiment_lookup(sent_dict, nextw, nextpos)

    return {'word': word, 'pos': pos, 'sentiment': sentiment,
            'prevpos': prevpos, 'prevtag': prevtag, 'prev_sentiment': prev_sentiment,
            'nextpos': nextpos, 'next_sentiment': next_sentiment}


def sentiment_lookup(dict, word, pos):
    """ Return a sentiment indicator for the given word with the given POS tag, according to the dictionary.
    Ignoring tags for now.
    Return values: positive, negative, neutral.
    """
    if word in dict['pos']:
        return 'positive'
    if word in dict['neg']:
        return 'negative'
    return 'neutral'


def train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', pickled=False):
    """Creates an 80/20 split of the examples in filename,
    trains the chunker on 80%, and evaluates the learned chunker on 20%.
    """
    if pickled:
        f = open(filename, 'rb')
        traind = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = traind['iob'][:split_size]
    test = traind['iob'][split_size:]
    posi_words = get_liu_lexicon(posit_lex_file)
    negi_words = get_liu_lexicon(nega_lex_file)
    chunker = ConsecutiveChunker(train, {'pos': posi_words, 'neg': negi_words})
    print chunker.evaluate(test)


def get_liu_lexicon(filename):
    """
    Return the list of sentiment words from the Liu-formatted sentiment lexicons (just a header then a list)
    """
    return [item.strip() for item in open(filename, "r").readlines() if re.match("^[^;]+\w",item)]


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
    train_and_test('restaurants-trial.xml','positive-words.txt', 'negative-words.txt')
