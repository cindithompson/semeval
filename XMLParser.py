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
        words = sentence.find('text').text
        #get rid of utf - no, do later
        ##words = words.encode('utf-8')
        a_terms = sentence.find('aspectTerms')
        sentiments = []
        terms = []
        froms, tos = [], []
        if a_terms is not None:
            for at in a_terms:
                terms.append(at.attrib['term'])
                if 'polarity' in at.attrib:
                    sentiments.append(at.attrib['polarity'])
                froms.append(int(at.attrib['from']))
                tos.append(int(at.attrib['to']))
        aspects.append(terms)
        #print "creating for sent:", words
        #print "terms:", terms
        seq, idxs = new_create_POS_ex(words, froms, tos)
        #print "seq:", seq
        #print "idx:", idxs
        ''' OLDER
        seq = create_POS_ex(words, froms, tos)
        idxs = find_start_ends(words, [t[0] for t in seq])
        '''
        examples.append(seq)
        indices.append(idxs)
        polarities.append(sentiments)
        text.append(words.strip())
    return {'orig': text, 'iob': examples, 'polarity': polarities, 'aspects': aspects,
            'id': ids, 'idx': indices}


def new_create_POS_ex(sent, froms, tos):
    """Create the IOB & POS tagged sequence
    """
    iob = []
    pos_tags = nltk.pos_tag(nltk.word_tokenize(sent.strip().encode('utf-8')))
    #print pos_tags
    tokens = [w for (w,t) in pos_tags]
    idxs = find_start_ends(sent, tokens)

    #print idxs
    curr_asp_idx = -1
    in_tag = False
    for i in range(len(pos_tags)):
        #print "processing: %s, in_tag: %s, at asp: %d" %(pos_tags[i], in_tag, curr_asp_idx)
        if in_tag:
            iob.append((pos_tags[i][0],pos_tags[i][1], 'I-Aspect'))
            if tos[curr_asp_idx] == idxs[i][1]:
                in_tag = False
        elif idxs[i][0] in froms:
            curr_asp_idx = froms.index(idxs[i][0])
            iob.append((pos_tags[i][0],pos_tags[i][1], 'B-Aspect'))
            if tos[curr_asp_idx] != idxs[i][1]:
                in_tag = True
        else:
            iob.append((pos_tags[i][0],pos_tags[i][1], 'O'))
    return iob, idxs


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
    #Following new for trial xml files, late march
    sent = sent.rstrip()
    while s_idx < len(sent):
        #print "S:%sS" %sent[s_idx]
        curr_term = terms[t_idx].decode('utf-8')
        #print "curr:%sS" %curr_term
        #print s_idx, curr_term
        #print "S:%sS" %sent[s_idx]
        #special case, quotes get changed
        if (curr_term == "''" or curr_term == '``') and sent[s_idx].startswith('"'):
            idxs.append((s_idx,s_idx+1))
            t_idx += 1
            s_idx += 1
        elif sent[s_idx:].startswith(curr_term):
            idxs.append((s_idx,s_idx+len(curr_term)))
            #print "found"
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
    return result


def create_xml(orig, iobs, ids, indices, sentiments=None, outfile='dump-answers.txt'):
    """Create the XML file for answer submission.
    Inputs:
    orig: text of original sentences
    iobs: the tagged aspect terms of each sentence
    ids: sentence ids
    outfile: name of file to write the XML to
    Returns: success indicator
    """
    f = codecs.open(outfile, mode='w', encoding='utf-8')
    f.write('<sentences>\n')
    for i in range(len(orig)):
        f.write('<sentence id=\"'+ids[i]+'\">\n')
        f.write('<text>'+orig[i]+'\n</text>\n')
        if len(iobs[i]) > 0:
            f.write('<aspectTerms>\n')
            if not sentiments:
                process_aspects(iobs[i], indices[i], ['neutral']*len(iobs[i]), f)
            else:
                process_aspects(iobs[i], indices[i], sentiments[i], f)
            f.write('</aspectTerms>\n')
        f.write('</sentence>\n')
    f.write('</sentences>\n')
    f.close()


def process_aspects(iob, idxs, sentiments, f, use_iobs=True):
    """Write the aspect term portion to the file.
    use_iobs indicates whether to use IOB representation or just BO.
    """
    if use_iobs:
        process_iob_aspects(iob, idxs, sentiments, f)
    else:
        process_bo_aspects(iob, idxs, sentiments, f)


#TODO is test this
def process_bo_aspects(bo, idxs, sentiments, f):
    in_term = False
    terms = []
    curr_sentiment = None
    senti_idx = 0
    start, end = 0, 0
    for i in range(len(iob)):
        if in_term:
            if iob[i][2].startswith('B'):
                if end < idxs[i][0]:
                    terms.append(' ' * (idxs[i][0]-end))
                end = idxs[i][1]
                terms.append(iob[i][0])
            else: #'O'
                in_term = False
                write_aterm(terms, start, end,  curr_sentiment, f)
        elif iob[i][2].startswith('B'):
            in_term = True
            start = idxs[i][0]
            end = idxs[i][1]
            terms = [iob[i][0]]
            curr_sentiment = sentiments[senti_idx]
            senti_idx += 1
    if in_term:
        write_aterm(terms, start, end, curr_sentiment, f)


def process_iob_aspects(iob, idxs, sentiments, f):
    in_term = False
    terms = []
    curr_sentiment = None
    senti_idx = 0
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
                write_aterm(terms, start, end, curr_sentiment, f)
                start = idxs[i][0]
                end = idxs[i][1]
                terms = [iob[i][0]]
                curr_sentiment = sentiments[senti_idx]
                senti_idx += 1
            else: #'O'
                in_term = False
                write_aterm(terms, start, end,  curr_sentiment, f)
        elif iob[i][2].startswith('B'):
            in_term = True
            start = idxs[i][0]
            end = idxs[i][1]
            terms = [iob[i][0]]
            curr_sentiment = sentiments[senti_idx]
            senti_idx += 1
    if in_term:
        write_aterm(terms, start, end, curr_sentiment, f)


def write_aterm(words, start, end, polarity, f):
    f.write('<aspectTerm term="')
    for w in words:
        w = w.decode('utf-8')
        f.write(w)
    f.write('" polarity="')
    f.write(polarity)
    f.write('" from="')
    f.write(str(start))
    f.write('" to="')
    f.write(str(end))
    f.write('"/>\n')


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
