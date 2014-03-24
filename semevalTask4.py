import pickle
import string
import XMLParser
import nltk
from nltk.stem.lancaster import LancasterStemmer
import re
import cPickle
from sklearn import cross_validation
import semeval_util


#globals for certain features
use_unk = True
stemming = True


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using maximum entropy; should also try NB
    """

    def __init__(self, train_sents, sent_dict):
        train_set = []
        self.vocab = semeval_util.vocabulary(train_sents)
        self.stemmer = LancasterStemmer()
        for tagged_sent in train_sents:
            #print "tagged", tagged_sent
            untagged_sent = nltk.tag.untag(tagged_sent)
            #print "untagged", untagged_sent
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = chunk_features(untagged_sent, i, sent_dict, history, self.vocab, self.stemmer)
                train_set.append((featureset, tag))
                history.append(tag)
        self.sent_dict = sent_dict

        #megam doesn't work in below - seems hard to get working; cg also doesn't work
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='iis',
            trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = chunk_features(sentence, i, self.sent_dict, history, self.vocab, self.stemmer)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


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
        #print "parse result:", tagged_sents
        return [(w, t, c) for ((w, t), c) in tagged_sents]

    def evaluate(self, gold):
        """ nltk machinery computes Acc, P,R, and F-measure for trees. But we are not using it!
        """
        #chunkscore = nltk.ChunkScore()
        results = []
        for tagged_sent in gold:
            #print "true:", tagged_sent
            guess = self.parse([(w,t) for (w,t,_c) in tagged_sent])
            #print "guess:", guess
            results.append(guess)
        #return chunkscore
        return results


def chunk_features(sentence, i, sent_dict, history, vocab, st):
    """ Get features for sentence at position i with history being tags seen so far.
    vocab is the top N% vocabulary words, and
    st is the stemmer.
    Returns: dictionary of features.
    """
    word, pos = sentence[i]
    sentiment = semeval_util.sentiment_lookup(sent_dict, word, pos)
    if use_unk and not word in vocab:
        word = "<UNK>"
    elif stemming:
        word = st.stem(word)
    objectivity = semeval_util.get_objectivity(sentiment)
    if i == 0:
        prevw, prevpos = "<START>", "<START>"
        prevtag = "<START>"
        prev_sentiment = "<START>"
        prev_obj = "<START>"
    else:
        prevw, prevpos = sentence[i-1]
        prevtag = history[i-1]
        prev_sentiment = semeval_util.sentiment_lookup(sent_dict, prevw, prevpos)
        if use_unk and not prevw in vocab:
            prevw = "<UNK>"
        elif stemming:
            prevw = st.stem(prevw)
        prev_obj = semeval_util.get_objectivity(prev_sentiment)

    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
        next_sentiment = "<END>"
        next_obj = "<END>"
    else:
        nextw, nextpos = sentence[i+1]
        next_sentiment = semeval_util.sentiment_lookup(sent_dict, nextw, nextpos)
        if use_unk and not nextw in vocab:
            nextw = "<UNK>"
        elif stemming:
            nextw = st.stem(nextw)
        next_obj = semeval_util.get_objectivity(next_sentiment)

    return {'word': word, 'pos': pos, 'sentiment': sentiment, 'obj': objectivity,
            'prevw': prevw, 'prevpos': prevpos, 'prevtag': prevtag, 'prev_sentiment': prev_sentiment, 'prev_obj': prev_obj,
            'nextw': nextw, 'nextpos': nextpos, 'next_sentiment': next_sentiment, 'next_obj': next_obj}


def train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt'):
    """Creates an 80/20 split of the examples in filename,
    trains the chunker on 80%, and evaluates the learned chunker on 20%.
    """
    traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = traind['iob'][:split_size]
    test = traind['iob'][split_size:]
    #Liu not in use for now
    #posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    #negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    chunker = ConsecutiveChunker(train, senti_dictionary)
    guessed_iobs = chunker.evaluate(test)
    semeval_util.compute_pr(test, guessed_iobs)
    #print chunker.evaluate(test)


def train_and_trial(trn_file, test_file, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', pickled=False):
    """ Train on the training file and test on the testing file
    """
    if pickled:
        f = open(trn_file, 'rb')
        traind = cPickle.load(f)
        f.close()
        f = open(test_file, 'rb')
        testd = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(trn_file)
        testd = XMLParser.create_exs(test_file)
    posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    chunker = ConsecutiveChunker(traind['iob'], senti_dictionary)
    print "done training"
    '''
    f = open('learned.pkl','wb')
    cPickle.dump(chunker,f)
    f.close()
    '''
    guessed_iobs = chunker.evaluate(testd['iob'])
    XMLParser.create_xml(testd['orig'],guessed_iobs,testd['id'],testd['idx'],'trial_answers.xml')
    semeval_util.compute_pr(testd['iob'], guessed_iobs)


def K_fold_train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', k=5, pickled=False):
    """Does K-fold cross-validation on the given filename
    """
    if pickled:
        f = open(filename, 'rb')
        traind = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
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
        #print test_set
        #print guesses
        r, p, f = semeval_util.compute_pr(test_set, guesses)
        tot_p += p
        tot_r += r
        tot_f1 += f
    print "ave Prec: %.2f, Rec: %.2f, F1: %.2f" %(tot_p/float(k), tot_r/float(k), tot_f1/float(k))


if __name__ == '__main__':
    train_and_test('restaurants-trial.xml','positive-words.txt', 'negative-words.txt')
