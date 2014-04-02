import pickle
import string
import XMLParser
import nltk
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
import re
import cPickle
from sklearn import cross_validation
import semeval_util
import argparse
import numpy as np


#globals for certain features
use_unk = True
stemming = True
use_dep_parse = False
use_svm = False
#indicates IOB encoding versus BO encoding of terms
#TODO is make this a class variable so can set from one place when prepping submissions
use_iob = False


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using maximum entropy; should also try NB
    """

    def __init__(self, train_sents, test_sents, sent_dict, dep_parses):
        train_set = []
        self.vocab = semeval_util.vocabulary(train_sents)
        self.stemmer = LancasterStemmer()
        idx = 0
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = chunk_features(untagged_sent, i, sent_dict, history, self.vocab, self.stemmer)
                if use_dep_parse:
                    dep_features = semeval_util.dep_features(untagged_sent, i, dep_parses[idx])
                    featureset.update(dep_features)
                if not use_iob:
                    if tag.startswith('I'):
                        tag = 'B-Aspect'
                train_set.append((featureset, tag))
                history.append(tag)
            idx += 1
        self.sent_dict = sent_dict

        if use_svm:
            print "creating vectorized"
            v_exs, self.v_test = create_vectorized(train_set, test_sents, sent_dict, self.vocab, self.stemmer)
            clfs = []
            print "training"
            for label in ['I', 'O', 'B']:
                clf = SVC(probability=True)
                exsx, lys = create_binary_exs(v_exs, train_set, label)
                clf.fit(exsx, lys)
                clfs.append(clf)
            self.classifier = clfs
        else:
            #megam doesn't work in below - seems hard to get working; cg also doesn't work
            self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='iis', trace=0)

    def tag(self, example):
        if use_svm:
            this_x = self.v_test[example]
            l_idx = 0
            labels = ['I','O','B']
            best_proba = 0
            label = 'O'
            for c in self.classifier:
                pos_idx = np.where(c.classes_ == 1.0)[0][0]
                this_p = c.predict_proba([this_x])
                #print "idx & probas:", pos_idx, this_p
                if this_p[0][pos_idx] > best_proba:
                    best_proba = this_p[0][pos_idx]
                    label = labels[l_idx]
                l_idx += 1
            return label
        else:
            sentence, dep_parse = example
            history = []
            for i, word in enumerate(sentence):
                featureset = chunk_features(sentence, i, self.sent_dict, history, self.vocab, self.stemmer)
                if use_dep_parse:
                    dep_features = semeval_util.dep_features(sentence, i, dep_parse)
                    featureset.update(dep_features)
                tag = self.classifier.classify(featureset)
                history.append(tag)
            return zip(sentence, history)


class ConsecutiveChunker(nltk.ChunkParserI):
    """
    Wrapper class for ConsecutiveChunkTagger. Note actual POS tagging is done
    before we even get here.
    """
    def __init__(self, train_sents, test_sents, sent_dict, dep_parses):
        """Creates the tagger (actually called parse) via a ML approach
        within CCTagger
        """
        tagged_sents = [[((w, t), c) for (w, t, c) in
                           sent]
                        for sent in train_sents]
        #print tagged_sents
        self.tagger = ConsecutiveChunkTagger(tagged_sents, test_sents, sent_dict, dep_parses)

    def parse(self, example):
        """ Return the "parse tree" version of an input POS-tagged sentence
         The leaves of the tree have tags for IOB labels
        """
        if use_svm:
            return self.tagger.tag(example)
        else:
            tagged_sents = self.tagger.tag(example)
            return [(w, t, c) for ((w, t), c) in tagged_sents]

    def evaluate(self, gold_examples):
        """ nltk machinery computes Acc, P,R, and F-measure for trees. But we are not using it!
        """
        results = []
        gold, dep_parses = gold_examples
        if use_svm:
            #never uses dep_parse for now
            v_idx = 0
            for tagged_sent in gold:
                ans = []
                for (w, t, _c) in tagged_sent:
                    ans.append((w, t, self.parse(v_idx)))
                    v_idx += 1
                results.append(ans)
            return results
        else:
            i = 0
            for tagged_sent in gold:
                #print "true:", tagged_sent
                if use_dep_parse:
                    guess = self.parse(([(w,t) for (w,t,_c) in tagged_sent], dep_parses[i]))
                else:
                    guess = self.parse(([(w,t) for (w,t,_c) in tagged_sent], []))
                i += 1
                #print "guess:", guess
                results.append(guess)
            return results


def create_vectorized(set1, set2, sent_dict, vocab, stemmer):
    n = len(set1)
    vec = DictVectorizer()
    set1_new = [ex for (ex, tag) in set1]
    set2_new = create_seq_exs(set2, sent_dict, vocab, stemmer)
    vectorized = vec.fit_transform(set1_new + set2_new).toarray()
    return vectorized[:n], vectorized[n:]


def create_seq_exs(gold, sent_dict, vocab, stemmer):
    results = []
    for tagged_sent in gold:
        untagged_sent = [(w,t) for (w,t,c) in tagged_sent]
        history = []
        for i, (word, pos, tag) in enumerate(tagged_sent):
            featureset = chunk_features(untagged_sent, i, sent_dict, history, vocab, stemmer)
            results.append(featureset)
            history.append('O') #dummy
    return results


def create_binary_exs(examples, orig_exs, label):
    resultX = []
    resulty = []
    for i in range(len(examples)):
        ex_v = examples[i]
        tag = orig_exs[i][1]
        resultX.append(ex_v)
        if tag.startswith(label):
            resulty.append(1.0)
        else:
            resulty.append(0.0)
    return resultX, resulty


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
            'prevw': prevw, 'prevpos': prevpos, #####'prevtag': prevtag,
            'prev_sentiment': prev_sentiment, 'prev_obj': prev_obj,
            'nextw': nextw, 'nextpos': nextpos, 'next_sentiment': next_sentiment, 'next_obj': next_obj}


def train_and_test(filename, parse_file, use_deps=False,
                   posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt'):
    """Creates an 80/20 split of the examples in filename,
    trains the chunker on 80%, and evaluates the learned chunker on 20%.
    """
    global use_dep_parse
    if use_deps:
        use_dep_parse = True
    traind = XMLParser.create_exs(filename)
    dep_parses = []
    if use_dep_parse:
        dep_parses = semeval_util.add_dep_parse_features(traind['iob'], parse_file, dictionary=True, iobs=True)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = traind['iob'][:split_size]
    test = traind['iob'][split_size:]
    test_deps = []
    if use_dep_parse:
        test_deps = dep_parses[split_size:]
    #Liu not in use for now
    #posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    #negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    chunker = ConsecutiveChunker(train, test, senti_dictionary, dep_parses)
    guessed_iobs = chunker.evaluate([test,test_deps])
    semeval_util.compute_pr(test, guessed_iobs)


def train_and_trial(trn_file, test_file, parse_file_train, parse_file_test, use_dep=False,
                    posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', pickled=False):
    """ Train on the training file and test on the testing file
    """
    global use_dep_parse
    if use_dep:
        use_dep_parse = True
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
    #posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    #negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    dep_parses = []
    if use_dep_parse:
        dep_parses = semeval_util.add_dep_parse_features(traind['iob'], parse_file_train, dictionary=True, iobs=True)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    chunker = ConsecutiveChunker(traind['iob'], testd['iob'], senti_dictionary, dep_parses)
    print "done training on %d examples" % len(traind['iob'])
    '''
    f = open('learned.pkl','wb')
    cPickle.dump(chunker,f)
    f.close()
    '''
    if use_dep_parse:
        dep_parses = semeval_util.add_dep_parse_features(traind['iob'], parse_file_test, dictionary=True, iobs=True)

    guessed_iobs = chunker.evaluate([testd['iob'], dep_parses])
    ###semeval_util.compute_pr(testd['iob'], guessed_iobs)
    return guessed_iobs


def K_fold_train_and_test(filename, parse_file, use_dep=False,
                          posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', k=5, pickled=False):
    """Does K-fold cross-validation on the given filename
    """
    global use_dep_parse
    if use_dep:
        print "using dependency parses"
        use_dep_parse = True
    if pickled:
        f = open(filename, 'rb')
        traind = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    dep_parses = traind['iob']
    if use_dep_parse:
        dep_parses = semeval_util.add_dep_parse_features(traind['iob'], parse_file, dictionary=True, iobs=True)
    #posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    #negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    senti_dictionary = semeval_util.get_mpqa_lexicon()
    kf = cross_validation.KFold(n, n_folds=k, indices=True)
    tot_p, tot_r, tot_f1 = 0, 0, 0
    for train, test in kf:
        print "next fold, split size: %d/%d" %(len(train), len(test))
        #print train
        train_set = []
        test_set = []
        train_parse = []
        test_parse = []
        for i in train:
            train_set.append(traind['iob'][i])
            train_parse.append(dep_parses[i])
        for i in test:
            test_set.append(traind['iob'][i])
            test_parse.append(dep_parses[i])
        chunker = ConsecutiveChunker(train_set, test_set, senti_dictionary, train_parse)
        guesses = chunker.evaluate([test_set, test_parse])
        #print test_set
        #print guesses
        r, p, f = semeval_util.compute_pr(test_set, guesses)
        tot_p += p
        tot_r += r
        tot_f1 += f
    print "ave Prec: %.2f, Rec: %.2f, F1: %.2f" %(tot_p/float(k), tot_r/float(k), tot_f1/float(k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="must be either lap or rest", type=str)
    parser.add_argument("-d", help="Use dependency parser",type=bool, default=False)
    parser.add_argument("-f", help="Number of folds", type=int, default=5)
    args = parser.parse_args()

    if args.task_name == 'rest':
        #train_and_trial('../PycharmProjects/emnlp/Rest_train_v2.pkl', '../PycharmProjects/emnlp/Rest_train_v2.pkl',
        #                '../PycharmProjects/emnlp/rest_train-parse.txt','../PycharmProjects/emnlp/rest_train-parse.txt',
        #                use_dep=args.d, pickled=True)
        K_fold_train_and_test('Rest_train_v2.pkl',
                              'rest_train-parse.txt', pickled=True, use_dep=args.d, k=args.f)
    elif args.task_name == 'lap':
        #train_and_trial('../PycharmProjects/emnlp/Laptop_train_v2.pkl', '../PycharmProjects/emnlp/Laptop_train_v2.pkl',
        #                '../PycharmProjects/emnlp/lap_Train-parse.txt','../PycharmProjects/emnlp/lap_Train-parse.txt',
        #                use_dep=args.d, pickled=True)

        K_fold_train_and_test('Laptop_train_v2.pkl',
                          'lap_Train-parse.txt',
                          pickled=True, use_dep=args.d, k=args.f)
        #print "getting the kinks out"

    elif args.task_name == 'dummy':
        K_fold_train_and_test('laptops-trial.pkl',
                              'lap-trial-parse.txt', pickled=True, use_dep=args.d, k=args.f)
    else:
        print "invalid task_name", args.task_name
