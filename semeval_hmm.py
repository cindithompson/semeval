import cPickle
import string
import XMLParser
import nltk
from nltk.tag import hmm
import re
from sklearn import cross_validation
import semeval_util
from nltk.stem.lancaster import LancasterStemmer


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using HMM
    """

    def __init__(self, train_sents, sent_dict):
        '''train_sents entries are in form [((w, pos_tag), iob_tag),...]
        '''
        train_set = []
        tag_set = []
        symbols = []
        self.stemmer = LancasterStemmer()
        self.just_pos = False
        self.use_pos = False
        for tagged_sent in train_sents:
            example = []
            for i, (wd_pos, tag) in enumerate(tagged_sent):
                tag_set.append(tag)
                if self.just_pos:
                    symb = wd_pos[1]
                elif self.use_pos:
                    #symb = wd_pos[0]+wd_pos[1]
                    symb = self.stemmer.stem(wd_pos[0]) + wd_pos[1]
                else:
                    symb = self.stemmer.stem(wd_pos[0])
                symbols.append(symb)
                example.append((symb, tag))
            train_set.append(example)
        trainer = hmm.HiddenMarkovModelTrainer(list(set(tag_set)), list(set(symbols)))
        self.hmm = trainer.train_supervised(train_set)

    def tag(self, sentence):
        #sentence is a list of (w,pos) tuples - create the features used during training
        example = []
        for w, pos in sentence:
            if self.just_pos:
                example.append(pos)
            elif self.use_pos:
                example.append(self.stemmer.stem(w) + pos)
            else:
                example.append(self.stemmer.stem(w))
        iob_tags = self.hmm.tag(example)
        result = []
        for i in range(len(sentence)):
            result.append((sentence[i], iob_tags[i][1]))
        return result


class ConsecutiveChunker(nltk.ChunkParserI):
    """
    Wrapper class for ConsecutiveChunkTagger. Note actual POS tagging is done
    before we even get here.
    """
    def __init__(self):
        pass

    def train(self, train_sents, sent_dict):
        """Creates the tagger (actually called parse) via a ML approach
        within CCTagger
        """
        tagged_sents = [[((w, t), c) for (w, t, c) in
                           sent]
                        for sent in train_sents]
        self.tagger = ConsecutiveChunkTagger(tagged_sents, sent_dict)

    def parse(self, sentence):
        """ Return the "parse tree" version of an input POS-tagged sentence
         The leaves of the tree have tags for IOB labels
        """
        tagged_sents = self.tagger.tag(sentence)
        #print "in parse:",tagged_sents
        return [(w, t, c) for ((w, t), c) in tagged_sents]

    def evaluate(self, gold):
        """ Doesn't actually evaluate in terms of scoring, but returns the
        sequences
        """
        results = []
        for tagged_sent in gold:
            guess = self.parse([(w,t) for (w,t,_c) in tagged_sent])
            results.append(guess)
        return results


def chunk_features(sentence, i, sent_dict, history):
    """ Get features for sentence at position i with history being tags seen so far.
    Returns: dictionary of features.
    """
    word, pos = sentence[i]
    sentiment = semeval_util.sentiment_lookup(sent_dict, word, pos)
    if i == 0:
        prevw, prevpos = "<START>", "<START>"
        prevtag = "<START>"
        prev_sentiment = "<START>"
        #prev_negtive = "<START>"
    else:
        prevw, prevpos = sentence[i-1]
        prevtag = history[i-1]
        prev_sentiment = semeval_util.sentiment_lookup(sent_dict, prevw, prevpos)
    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
        next_sentiment = "<END>"
    else:
        nextw, nextpos = sentence[i+1]
        next_sentiment = semeval_util.sentiment_lookup(sent_dict, nextw, nextpos)

    return {'word': word, 'pos': pos, 'sentiment': sentiment,
            'prevpos': prevpos, 'prevtag': prevtag, 'prev_sentiment': prev_sentiment,
            'nextpos': nextpos, 'next_sentiment': next_sentiment}


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
    #posi_words = get_liu_lexicon(posit_lex_file)
    #negi_words = get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    chunker = ConsecutiveChunker()
    chunker.train(train, senti_dictionary)
    guessed_iobs = chunker.evaluate(test)
    semeval_util.compute_pr(test, guessed_iobs)


def K_fold_train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', k=2, pickled=False):
    """Does K-fold cross-validation on the given filename
    """
    if pickled:
        f = open(filename, 'rb')
        traind = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    #posi_words = get_liu_lexicon(posit_lex_file)
    #negi_words = get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    kf = cross_validation.KFold(n, n_folds=k, indices=True)
    tot_p, tot_r, tot_f1 = 0, 0, 0
    for train, test in kf:
        print "next fold, split size: %d/%d" %(len(train), len(test))
        #print train
        train_set = []
        test_set = []
        for i in train:
            train_set.append(traind['iob'][i])
        for i in test:
            test_set.append(traind['iob'][i])
        chunker = ConsecutiveChunker(train_set, senti_dictionary)
        guesses = chunker.evaluate(test_set)
        print test_set
        print guesses
        r, p, f = semeval_util.compute_pr(test_set, guesses)
        tot_p += p
        tot_r += r
        tot_f1 += f
    print "ave Prec: %.2f, Rec: %.2f, F1: %.2f" %(tot_p/float(k), tot_r/float(k), tot_f1/float(k))


if __name__ == '__main__':
    train_and_test('restaurants-trial.xml','positive-words.txt', 'negative-words.txt')
