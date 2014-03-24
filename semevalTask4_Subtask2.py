import string
import XMLParser
import nltk
from collections import defaultdict
import re
import cPickle
import semeval_util


class ConsecutiveChunkTagger(nltk.TaggerI):
    """
    Trains using maximum entropy; should also try NB
    """

    def __init__(self, train_sents, sent_dict, senti_tags):
        """Creates the clf via a ML approach .
        Inputs: train_sents: tuples of (iob-tagged sentence, aspect label list)
        sent_dict: sentiment dictionary, positive and negative
        senti_tags: the sentiment tag for each train_sent
        """
        train_set = []
        n = len(train_sents)
        print "%d training examples" % n
        for x in range(n):
            tagged_sent = train_sents[x][0]
            #these are really sentiment tags for the aspect terms, not the IOB tags as in extraction task
            aspect_tags = train_sents[x][1]
            print "x:", tagged_sent, aspect_tags
            n_aspect = 0
            sentiment = senti_tags[x]
            print "sentim:",sentiment
            history = []
            # create a training example for each aspect term
            for i, (_word, _pos, tag) in enumerate(tagged_sent):
                #these will get thrown away except for in the nbhd of aspect terms
                feature_set = chunk_features(tagged_sent, i, sent_dict, history)
                if tag.startswith('B'):
                    # TODO is more experiments on larger set to see which of the below makes most sense
                    #feature_set['hi_lvl_pos_senti'] = sentiment[0]
                    #feature_set['hi_lvl_neg_senti'] = sentiment[1]
                    feature_set['hi_lvl_senti'] = sentiment[0]-sentiment[1]
                    #feature_set['hi_lvl_senti'] = sentiment[0] > sentiment[1]
                    feature_set['num_aspects'] = len(aspect_tags)
                    feature_set.update(get_head_features(tagged_sent, i, sent_dict))
                    train_set.append((feature_set, aspect_tags[n_aspect]))
                    history.append(aspect_tags[n_aspect])
                    #print "adding example:", (feature_set, aspect_tags[n_aspect])
                    n_aspect += 1

        self.sent_dict = sent_dict
        #megam doesn't work in below - seems hard to get working; cg also doesn't work
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='iis',
            trace=0)

    def parse(self, sentence):
        words, sentiment = sentence
        #print "in tag with:", sentence
        #count up # of aspect terms
        num_aspects = 0
        for (_w, _p, tag) in words:
            if tag.startswith('B'):
                num_aspects += 1
        history = []
        for i, (_word, _pos, tag) in enumerate(words):
            feature_set = chunk_features(words, i, self.sent_dict, history)
            if tag.startswith('B'):
                #feature_set['hi_lvl_pos_senti'] = sentiment[0]
                #feature_set['hi_lvl_neg_senti'] = sentiment[1]
                feature_set['hi_lvl_senti'] = sentiment[0] - sentiment[1]
                feature_set['num_aspects'] = num_aspects
                feature_set.update(get_head_features(words, i, self.sent_dict))
                label = self.classifier.classify(feature_set)
                history.append(label)
        return history

    def evaluate(self, gold):
        """ get sentiment labeling accuracy
        """
        n = 0
        n_corr = 0
        #temp
        all_labels = []
        for tagged_sent, senti_label in gold:
            #print "one gold:", tagged_sent
            #print "senti:", senti_label
            sentence, label = tagged_sent
            result = self.parse((sentence,senti_label))
            #print "result:", result
            #print "same number of results as labels?", len(result) == len(label)
            for i in range(len(result)):
                n += 1
                all_labels.append(label[i])
                if result[i] == label[i]:
                    n_corr += 1
        counts = defaultdict(int)
        for l in all_labels:
            counts[l] += 1
        print counts
        if n > 0:
            return float(n_corr)/n
        return 0


def get_head_features(sentence, i, sent_dict):
    """
    Get features associated with the aspect term - mostly looking at the head word
    of the phrase, assuming (for now) it's the last word in the phrase.
    Input: sentence: (w,pos,iob) sequence
    i: position of first aspect term of the phrase, within the sentence
    sent_dict: polarity lexicon

    Returns: Dictionary containing: POS tag of head of phrase, head word of phrase,
    sentiment of head word, word/POS/sentiment after the whole phrase. (Note we already have the
    word/POS/sentiment before the whole phrase from the chunk_features method).
    """
    word, pos, iob = sentence[i]
    #print word, pos, iob, i
    #print sentence
    #find head
    found_head = False
    while not found_head:
        if len(sentence) > i+1:
            n_w, n_pos, n_iob = sentence[i+1]
            if n_iob.startswith('O') or n_iob.startswith('B'):
                found_head = True
            else:
                i += 1
                word, pos, iob = sentence[i]
        else:
            found_head = True
    sentiment = sentiment_lookup(sent_dict, word, pos)
    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
        next_sentiment = "<END>"
        next_tag = "<END>"
    else:
        nextw, nextpos, next_tag = sentence[i+1]
        next_sentiment = sentiment_lookup(sent_dict, nextw, nextpos)
    return {'word': word, 'pos': pos, #'iob': iob,
            'sentiment': sentiment, 'nextw': nextw, 'nextpos': nextpos,
            'nextiob': next_tag, 'next_sentiment': next_sentiment}


def chunk_features(sentence, i, sent_dict, history):
    """ Get features for sentence at position i.
    Returns: dictionary of features.
    """
    word, pos, iob = sentence[i]
    sentiment = sentiment_lookup(sent_dict, word, pos)
    if i == 0:
        prevw, prevpos = "<START>", "<START>"
        prevtag = "<START>"
        prev_sentiment = "<START>"
    else:
        prevw, prevpos, prevtag = sentence[i-1]
        prev_sentiment = sentiment_lookup(sent_dict, prevw, prevpos)
    if len(history) > 0:
        prev_class = history[-1]
    else:
        prev_class = "<START>"
    if i == len(sentence)-1:
        nextw, nextpos = "<END>", "<END>"
        next_sentiment = "<END>"
        next_tag = "<END>"
    else:
        nextw, nextpos, next_tag = sentence[i+1]
        next_sentiment = sentiment_lookup(sent_dict, nextw, nextpos)

    return {'word': word, 'pos': pos, #'iob': iob,
            'sentiment': sentiment,
            'prev_classification': prev_class,
            'prevwd': prevw, 'prevpos': prevpos, 'previob': prevtag, 'prev_sentiment': prev_sentiment,
            'nextw': nextw, 'nextpos': nextpos, 'nextiob': next_tag, 'next_sentiment': next_sentiment}


negateWords = ["won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't",
               "isn't" "aren't" "wasn't" "weren't" "hasn't" "haven't" "hadn't" "doesn't" "don't" "didn't",
               "not", "no", "never"]


def senti_classify(sentence, pdict, ndict):
    """ Simple sentence-level sentiment classifier using sentiment lexicons that are
    lists of words, and count up pos/neg words plus incorporate simple negation
    """
    pcount = 0
    ncount = 0
    words = re.split("(\W+)", sentence.lower())

    #flips the sentiment when negation words are seen
    modifier = 1
    for w in words:
        if w in negateWords:
            modifier = -1
        #reset negation upon punctuation
        if w.find(".") >= 0 or w.find(",") >= 0 or w.find(";") >= 0 or w.find("--") >= 0:
            modifier = 1
        if w in pdict:
            pcount += modifier
        if w in ndict:
            ncount += modifier
    return pcount, ncount


def train_and_test(filename, posit_lex_file='positive-words.txt', nega_lex_file='negative-words.txt', pickled=False):
    """Creates an 80/20 split of the examples in filename,
    trains the sentiment classifier on 80%, and evaluates the learned classifier on 20%.
    """
    if pickled:
        f = open(filename, 'rb')
        traind = cPickle.load(f)
        f.close()
    else:
        traind = XMLParser.create_exs(filename)
    n = len(traind['iob'])
    split_size = int(n * 0.8)
    train = zip(traind['iob'][:split_size], traind['polarity'][:split_size])
    test = zip(traind['iob'][split_size:], traind['polarity'][split_size:])
    posi_words = semeval_util.get_liu_lexicon(posit_lex_file)
    negi_words = semeval_util.get_liu_lexicon(nega_lex_file)
    full_senti_label = [senti_classify(sentence, posi_words, negi_words) for sentence in traind['orig']]

    chunker = ConsecutiveChunkTagger(train, {'pos': posi_words, 'neg': negi_words}, full_senti_label)
    print chunker.evaluate(zip(test, full_senti_label[split_size:]))


if __name__ == '__main__':
    #f = file('obj.save', 'wb')
    #cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()
    #train_and_test('restaurants-trial.xml', 'positive-words.txt', 'negative-words.txt')
    f = open('../PycharmProjects/emnlp/Rest_train_v2.pkl', 'rb')
    traind = cPickle.load(f)
    f.close()

    posi_words = semeval_util.get_liu_lexicon('positive-words.txt')
    negi_words = semeval_util.get_liu_lexicon('negative-words.txt')
    full_senti_label = [senti_classify(sentence, posi_words, negi_words) for sentence in traind['orig']]
    #split_size = int(len(traind['orig']) * .25)
    split_size = len(traind['orig'])
    #subset = zip(traind['iob'][:split_size], traind['polarity'][:split_size])
    tp, tneg, tneutr = 0., 0., 0.
    fnn = 0.
    missed_neut, fpn = 0., 0.
    wrong_empties = 0.
    for i in range(split_size):
        labels = traind['polarity'][i]
        if len(labels) > 0:
            senti = 0
            for l in labels:
                if l=='positive':
                    senti += 1
                elif l == 'negative':
                    senti -= 1
            if senti > 0:
                if full_senti_label[i] > 0:
                    tp += 1
                else:
                    fnn += 1
            elif senti == 0:
                if full_senti_label[i] == 0:
                    tneutr += 1
                else:
                    missed_neut += 1
                    print traind['orig'][i]
                    print traind['polarity'][i]
                    print traind['aspects'][i]
            else: # senti < 0
                print
                if full_senti_label[i] < 0:
                    tneg += 1
                else:
                    fpn += 1
        else:
            if full_senti_label[i] != 0:
                wrong_empties += 1
                print "wrong empty, senti:", full_senti_label[i]
                print traind['orig'][i]
                print traind['polarity'][i]
                print traind['aspects'][i]

    print "TP: %f, TN: %f, TNeut: %f" %(tp/split_size, tneg/split_size, tneutr/split_size)
    print "FP: %f, FN: %f, FNeut: %f" %(fnn/split_size, missed_neut/split_size, fpn/split_size)
    print "wrong empties: ", wrong_empties/split_size


