__author__ = 'Steven'
import xml.etree.ElementTree as ET
import pickle
import string
import nltk
import re
import codecs


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
    #needed for final dump of answers to XML
    ids = []
    #also needed is original positions, too hard to get back later
    indices = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        ids.append(sentence.attrib['id'])
        words = sentence.find('text').text.strip()
        a_terms = sentence.find('aspectTerms')
        sentiments = []
        terms = []
        froms, tos = [], []
        if a_terms is not None:
            for at in a_terms:
                terms.append(at.attrib['term'])
                sentiments.append(at.attrib['polarity'])
                froms.append(int(at.attrib['from']))
                tos.append(int(at.attrib['to']))
        aspects.append(terms)
        #print "creating for sent:", words
        #print "terms:", terms
        seq = create_POS_ex(words, froms, tos)
        examples.append(seq)
        idxs = find_start_ends(words, [t[0] for t in seq])
        indices.append(idxs)
        polarities.append(sentiments)
        text.append(words)
    return {'orig': text, 'iob': examples, 'polarity': polarities, 'aspects': aspects,
            'id': ids, 'idx': indices}


def create_iob(sent, froms, tos):
    """ Support method for creating an example. Just get the IOB tags from
    character indexes in the original sentence.
    """
    words = sent.split()
    cidx = 0
    to = 0
    iob = []
    in_tag = False
    for w in words:
        if in_tag:
            if cidx < to:
                iob.append((w, 'I-Aspect'))
            else:
                iob.append((w, 'O'))
                in_tag = False
        elif cidx in froms:
            in_tag = True
            iob.append((w, 'B-Aspect'))
            to = tos[froms.index(cidx)]
        else:
            iob.append((w, 'O'))
        cidx += len(w) + 1
    return iob


def find_start_ends(sent, terms):
    """ Find the position in sent of each of the terms (which
    might be words but might be a substring of the word because of punctuation splits
    that tokenizing did).
    Return: a list of (start, end) tuples, one per term. NOTE: the end tuple
    uses the traditional python convention of pointing one char past the term.
    """
    idxs = []
    t_idx, s_idx = 0,0
    while s_idx < len(sent):
        curr_term = terms[t_idx]
        #print s_idx, curr_term
        #print sent[s_idx:]
        #special case, quotes get changed
        if (curr_term == "''" or curr_term == '``') and sent[s_idx].startswith('"'):
            idxs.append((s_idx,s_idx+1))
            t_idx += 1
            s_idx += 1
        elif sent[s_idx:].startswith(curr_term):
            idxs.append((s_idx,s_idx+len(curr_term)))
            t_idx += 1
            s_idx += len(curr_term)
        else:
            s_idx += 1
    return idxs


def create_POS_ex(sent, froms, tos):
    """Create the POS-tagged IOB part of the example.
    Input: sent: original text of sentence;
    froms: start indexes of aspect terms
    tos: end indexes of aspect terms in sent
    """
    iob = create_iob(sent, froms, tos)
    pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
    #print iob
    #print pos_tags
    result = []
    i_idx = 0
    for w, ptag in pos_tags:
        #print w, i_idx, iob[i_idx]
        #there can be punctuation at the end of a sentence so i_idx can get to large
        if i_idx < len(iob) and iob[i_idx][0].startswith(w):
            result.append((w, ptag, iob[i_idx][1]))
            i_idx += 1
        else:
            result.append((w, ptag, 'O'))
    # new code as of mid-March, getting the original character positions of the IOBs
    #print result
    #print find_start_ends(sent, [t[0] for t in result])
    return result


def create_xml(orig, iobs, ids, indices, outfile='dump-answers.txt'):
    """Create the XML file for answer submission.
    Inputs:
    orig: text of original sentences
    iobs: the tagged aspect terms of each sentence
    ids: sentence ids
    outfile: name of file to write the XML to
    Returns: success indicator
    """
    #f = open(outfile, 'w')
    f = codecs.open(outfile, mode='w', encoding='utf-8')
    f.write('<sentences>\n')
    for i in range(len(orig)):
        f.write('<sentence id=\"'+ids[i]+'\">\n')
        f.write('<text>\n'+orig[i]+'\n</text>\n')
        if len(iobs[i]) > 0:
            f.write('<aspectTerms>\n')
            process_aspects(iobs[i], indices[i], f)
            f.write('</aspectTerms>\n')
        f.write('</sentence>\n')
    f.write('</sentences>\n')
    f.close()


def process_aspects(iob, idxs, f):
    in_term = False
    terms = []
    start, end = 0, 0
    #print "idxs:" ,idxs
    for i in range(len(iob)):
        if in_term:
            if iob[i][2].startswith('I'):
                #previous end less than current start
                if end < idxs[i][0]:
                    terms.append(' ' * (idxs[i][0]-end))
                end = idxs[i][1]
                terms.append(iob[i][0])
            elif iob[i][2].startswith('B'):
                write_aterm(terms, start, end, f)
                start = idxs[i][0]
                end = idxs[i][1]
                terms = [iob[i][0]]
            else: #'O'
                in_term = False
                write_aterm(terms, start, end, f)
        elif iob[i][2].startswith('B'):
                in_term = True
                start = idxs[i][0]
                end = idxs[i][1]
                terms = [iob[i][0]]
    if in_term:
        write_aterm(terms, start, end, f)


def write_aterm(words, start, end, f):
    f.write('<aspectTerm term="')
    for w in words:
        f.write(w)
    f.write('" polarity="neutral" from="')
    f.write(str(start))
    f.write('" to="')
    f.write(str(end))
    f.write('"/>\n')


def create_POS_ex_old(sentence, words):
    """ OLDER VERSION: assumes each aspect term occurs once.
    Create the POS-tagged IOB part of the example.
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


def dbg(inpt):
    print inpt


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

    examples = create_exs('restaurants-trial.xml')
    create_xml(examples['orig'],examples['iob'],examples['id'],examples['idx'])
