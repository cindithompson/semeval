import XMLParser
import task4_stask2
import semeval_util
import argparse
import cPickle
import sys

#dummy trains and tests on the same data just to debug this harness
names = ['lap', 'rest', 'dummy']
pickle_trains = ['Laptop_train_v2.pkl',    'Rest_train_v2.pkl',                    'laptops-trial.pkl']
pickle_tests = ['Laptops_Test_Data_phaseB.pkl', 'Restaurants_Test_Data_phaseB.pkl', 'laptops-trial.pkl']
parses_trains = ['lap_Train-parse.txt',    'rest_train-parse.txt',                  'lap-trial-parse.txt']
parses_tests = ['laptops_test_phaseA-parse.txt','rest_test_phaseA-parse.txt',       'lap-trial-parse.txt']
results_files = ['lap_phaseB.xml',         'rest_phaseB.xml',                       'lap-trial_phaseB.xml']


def get_data(dataset_name):
    idx = names.index(dataset_name)
    return pickle_trains[idx], pickle_tests[idx], parses_trains[idx], parses_tests[idx], results_files[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="must be either lap or rest or dummy", type=str)
    #later if time
    parser.add_argument("-p", help="Specify that train_file is an already learned clf",type=bool, default=False)
    parser.add_argument("-dep", help="If true, use dependency parse features", type=bool, default=False)
    args = parser.parse_args()

    train_file, test_file, parse_train_file, parse_test_file, out_xml_file = get_data(args.task_name)

    print "IGNORING PARSE args if given"

    baseline = False
    if baseline:
        print "WARNING, using baseline"
        f = open(train_file, 'rb')
        traind = cPickle.load(f)
        f.close()
        senti_dictionary = semeval_util.get_mpqa_lexicon()
        negate_wds = semeval_util.negateWords
        results = []
        for iob in traind['iob']:
            polarities = semeval_util.create_sentiment_sequence(iob, senti_dictionary, negate_wds)
            translated = []
            for p, n in polarities:
                if p > n:
                    translated.append('positive')
                elif n > p:
                    translated.append('negative')
                else:
                    translated.append('neutral')
            results.append(translated)
        semeval_util.compute_sent_acc(traind['polarity'], results)
        XMLParser.create_xml(traind['orig'], traind['iob'], traind['id'], traind['idx'], sentiments=results,
                             outfile='baseline.xml')
        sys.exit()
    else:
        results = task4_stask2.train_and_trial(train_file, test_file)

    #create results file
    f = open(test_file, 'rb')
    testd = cPickle.load(f)
    f.close()
    XMLParser.create_xml(testd['orig'], testd['iob'], testd['id'], testd['idx'], sentiments=results, outfile=out_xml_file)





