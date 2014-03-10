__author__ = 'Steven'
import xml.etree.ElementTree as ET
import pickle
import string
import nltk


def pickle_laptop_data(filename):
    """
    handle the laptop training dataset, pickle the terms into the file 'Laptop_term.txt'
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    #output the root tag
    #print 'root tag:'+root.tag #is sentences

    Laptop_term = []
    #output the children nodes of root
    for child in root:
        for aspect_term in child[-1]:
            if aspect_term.attrib['term'] not in Laptop_term: #aspect_term.tag == 'term' and
                #print aspect_term.tag, aspect_term.attrib
                Laptop_term.append(aspect_term.attrib['term'])
    with open('Laptop_term.txt', 'wb') as handle:
        pickle.dump(Laptop_term, handle)

    #with open('Laptop_term.txt', 'rb') as handle:
    #    print pickle.loads(handle.read())


#the term, polarity, from, and to are attrib's of the aspectTerm elements
#version that just assumes a single occ of each aspect term (or tags all seen as part of aspect term)
def create_exs(filename):
    """ Create a set of training examples from a semeval XML file
    """
    examples = []
    polarities = []
    aspects = []
    text = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        words = sentence.find('text').text.strip()
        a_terms = sentence.find('aspectTerms')
        sentiments = []
        terms = []
        if a_terms is not None:
            for at in a_terms:
                terms.append(at.attrib['term'])
                sentiments.append(at.attrib['polarity'])
        aspects.append(terms)
        #print "creating for sent:", words
        #print "terms:", terms
        seq = create_POS_ex(words, terms)
        examples.append(seq)
        polarities.append(sentiments)

        text.append(words)
    return {'orig': text, 'iob': examples, 'polarity': polarities, 'aspects': aspects}


def create_POS_ex(sentence, words):
    """ Create the POS-tagged IOB part of the example.
    Input: sentence: original text of sentence;
    words: aspect phrases in the sentence
    """
    #split aspect phrases in words into pieces
    starts, continuations = split_words(words)
    pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
    iob = []
    in_tag = False
    for w, ptag in pos_tags:
        if in_tag:
            if w in continuations:
                iob.append((w, ptag, 'I-Aspect'))
            else:
                in_tag = False
                iob.append((w, ptag, 'O'))
        elif w in starts:
            iob.append((w, ptag, 'B-Aspect'))
            in_tag = True
        else:
            iob.append((w, ptag, 'O'))
    return iob

def split_words(words):
    """Given a phrase, return a tuple: the first word in the phrase (as a list),
    and a list of the "tail" of the phrase, if any
    """
    first_wds = []
    rest_wds = []
    for word in words:
        tokens = word.split()
        first_wds.append(tokens[0])
        rest_wds.extend(tokens[1:])
    return first_wds, rest_wds


def create_exs_older(filename):
    """ Create a set of training examples from a semeval XML file
    """
    examples = []
    polarities = []
    text = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        #words = sentence[0].text #assumes text comes first
        words = sentence.find('text').text.strip()
        a_terms = sentence.find('aspectTerms')
        froms, tos = [], []
        sentiments = []
        if a_terms is not None:
            for at in a_terms:
                froms.append(int(at.attrib['from']))
                tos.append(int(at.attrib['to']))
                sentiments.append(at.attrib['polarity'])
        ###seq = create_one_ex(words, froms, tos)
        seq = create_one_withPOS(words, froms, tos)
        examples.append(seq)
        polarities.append(sentiments)
        text.append(words)
    return {'orig': text, 'iob': examples, 'polarity': polarities}


#older version without POS tags
def create_one_ex(sent, froms, tos):
    """ given a sentence and the positions of the aspect terms in the sentence,
    return a list of word/tag pairs, where the tags are IOB indicating in, begin, or out of a aspect term
    Also strips punctuation
    """
    words = sent.split()
    cidx = 0
    iob = []
    it = iter(words)
    for w in it:
        if cidx in froms:
            iob.append(create_seq_ele(w, 'B-Aspect'))
            a_idx = froms.index(cidx)
            to = tos[a_idx]
            cidx += len(w) + 1
            while cidx < to:
                wd = it.next()
                iob.append(create_seq_ele(wd, 'I-Aspect'))
                cidx += len(wd) + 1
        else:
            iob.append(create_seq_ele(w, 'O'))
            cidx += len(w) + 1
    #print "final IOB", iob
    return iob


#doesn't work with punctuation, putting on back burner
def create_one_withPOS(sent, froms, tos):
    pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
    cidx = 0
    iob = []
    it = iter(pos_tags)
    for w,ptag in it:
        if cidx in froms:
            iob.append((w, ptag, 'B-Aspect'))
            a_idx = froms.index(cidx)
            to = tos[a_idx]
            cidx += len(w) + 1
            while cidx < to:
                wd,ptag = it.next()
                iob.append((wd, ptag, 'I-Aspect'))
                cidx += len(wd) + 1
        else:
            iob.append((w, ptag, 'O'))
            cidx += len(w) + 1
    #print "final IOB", iob
    return iob


def create_seq_ele(word, tag):
    """ Dummy POS tags for now
    """
    return word.strip(string.punctuation), 'dummy', tag


def pickle_restaurants_dataset(filename):
    """
    handle the restaurants' training dataset and pickle the terms into the file 'restaurants_term.txt'
    the categories was pickled to the file 'restaurants_categories.txt'
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    #output the root tag
    #print 'root tag:'+root.tag

    restaurants_term = []
    restaurants_categories = []
    #output the children nodes of root
    for child in root:
        for aspect_term in child[-1]:
            if aspect_term.attrib['category'] not in restaurants_categories: #aspect_term.tag == 'term' and
            #print aspect_term.tag, aspect_term.attrib
                restaurants_categories.append(aspect_term.attrib['category'])

        for aspect_term in child[-2]:
            if aspect_term.attrib['term'] not in restaurants_term: #aspect_term.tag == 'term' and
            #print aspect_term.tag, aspect_term.attrib
                restaurants_term.append(aspect_term.attrib['term'])
    #print restaurants_term
    #print restaurants_categories

    with open('restaurants_term.txt', 'wb') as handle:
        pickle.dump(restaurants_term, handle)
    with open('restaurants_categories.txt', 'wb') as handle:
        pickle.dump(restaurants_categories, handle)
    #with open('restaurants_term.txt', 'rb') as handle:
    #    print pickle.loads(handle.read())
    #with open('restaurants_categories.txt', 'rb') as handle:
    #    print pickle.loads(handle.read())


def term2categories():
    """
    show the relation between the aspect and the categories
    """
    term2categories={'service': [], 'food': [],  'anecdotes/miscellaneous': [], 'price': [], 'ambience': []}
    tree = ET.parse('Restaurants_Train.xml')
    root = tree.getroot()

    for child in root:
        for aspect_category in child[-1]:
            for aspect_term in child[-2]:
                term2categories[aspect_category.attrib['category']] .append(aspect_term.attrib['term'])
    for item in term2categories.keys():
        print item, term2categories[item]
    #print restaurants_term
    #print restaurants_categories


def show_term():
    """
    show the result of parsing the xml file
    """
    with open('Laptop_term.txt', 'rb') as handle:
        print 'Laptop_term:'
        print pickle.loads(handle.read())
    with open('restaurants_term.txt', 'rb') as handle:
        print 'restaurants_term:'
        print pickle.loads(handle.read())
    with open('restaurants_categories.txt', 'rb') as handle:
        print 'restaurants_categories:'
        print pickle.loads(handle.read())


if __name__ == '__main__':
    #pickle_laptop_data('Laptops_Train.xml')
    #pickle_restaurants_dataset('Restaurants_Train.xml')
    #show_term()
    #term2categories()

    print create_exs('restaurants-trial.xml')
