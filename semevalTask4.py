import pickle
import string
import XMLParser
import nltk
import scipy
from nltk.corpus import conll2000
from nltk.chunk.util import conlltags2tree
import re
import cPickle
from sklearn import cross_validation


#We are not using this, it's just an example to play around with chunking from the NLTK book
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

    def __init__(self, train_sents, sent_dict):
        train_set = []
        for tagged_sent in train_sents:
            #print "tagged", tagged_sent
            untagged_sent = nltk.tag.untag(tagged_sent)
            #print "untagged", untagged_sent
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = chunk_features(untagged_sent, i, sent_dict, history)
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
            featureset = chunk_features(sentence, i, self.sent_dict, history)
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
        #conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        #print "conlltags:", conlltags
        #return conlltags2tree(conlltags)

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
            #score thinks things should be in trees
            #chunkscore.score(conlltags2tree(tagged_sent), self.parse([(w,t) for (w,t,_c) in tagged_sent]))
        #return chunkscore
        return results


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


def train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt'):
    """Creates an 80/20 split of the examples in filename,
    trains the chunker on 80%, and evaluates the learned chunker on 20%.
    """
    traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = traind['iob'][:split_size]
    test = traind['iob'][split_size:]
    posi_words = get_liu_lexicon(posit_lex_file)
    negi_words = get_liu_lexicon(nega_lex_file)
    chunker = ConsecutiveChunker(train, {'pos': posi_words, 'neg': negi_words})
    guessed_iobs = chunker.evaluate(test)
    compute_pr(test, guessed_iobs)
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
    posi_words = get_liu_lexicon(posit_lex_file)
    negi_words = get_liu_lexicon(nega_lex_file)
    chunker = ConsecutiveChunker(traind['iob'], {'pos': posi_words, 'neg': negi_words})
    print "done training"
    f = open('learned.pkl','wb')
    cPickle.dump(chunker,f)
    f.close()
    guessed_iobs = chunker.evaluate(testd['iob'])
    XMLParser.create_xml(testd['orig'],guessed_iobs,testd['id'],testd['idx'],'trial_answers.xml')
    compute_pr(testd['iob'], guessed_iobs)


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
    posi_words = get_liu_lexicon(posit_lex_file)
    negi_words = get_liu_lexicon(nega_lex_file)
    kf = cross_validation.KFold(n, n_folds=k, indices=True)
    for train, test in kf:
        print "next fold, split size: %d/%d" %(len(train), len(test))
        #print train
        train_set = []
        test_set = []
        for i in train:
            train_set.append(traind['iob'][i])
        for i in test:
            test_set.append(traind['iob'][i])
        chunker = ConsecutiveChunker(train_set, {'pos': posi_words, 'neg': negi_words})
        guesses = chunker.evaluate(test_set)
        #print test_set
        #print guesses
        compute_pr(test_set, guesses)


def compute_pr(c_iobs, g_iobs):
    """Compute precision, recall, and F1
    """
    tp = 0
    fp = 0
    n = 0
    print "num examples:", len(c_iobs)
    #c_iobs = corrects['iob']
    #g_iobs = guesses['iob']
    for i in range(len(c_iobs)):
        one_corr = c_iobs[i]
        one_guess = g_iobs[i]
        #just counting the first word in each aspect phrase
        begin_terms = [x[0] for x in one_corr if x[2].startswith('B')]
        n += len(begin_terms)
        #print "n after %s: %d" %(one_corr, n)
        #print "corr", begin_terms
        for j in range(len(one_guess)):
            if one_guess[j][2].startswith('B'):
                #print "guess", one_guess[j]
                if one_guess[j][0] in begin_terms:
                    tp += 1
                else:
                    fp += 1
    recall = float(tp)/n
    if (tp+fp)>0:
        precision = float(tp)/(tp + fp)
    else:
        precision = 1.0
    if tp > 0:
        f1 = (2 * precision * recall)/(precision + recall)
    else:
        f1 = 0
    print "Recall %f0.2, Precision: %f0.2, F1 %f0.2" %(recall, precision, f1)


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
