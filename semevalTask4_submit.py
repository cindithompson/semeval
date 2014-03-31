import XMLParser
import semevalTask4
import argparse
import cPickle

#dummy trains and tests on the same data just to debug this harness
names = ['lap', 'rest', 'dummy']
pickle_trains = ['Laptop_train_v2.pkl','Rest_train_v2.pkl','laptops-trial.pkl']
pickle_tests = ['Laptops_test_phaseA.pkl','Restaurants_test_phaseA.pkl','laptops-trial.pkl']
parses_trains = ['lap_Train-parse.txt','rest_train-parse.txt','lap-trial-parse.txt']
parses_tests = ['laptops_test_phaseA-parse.txt','rest_test_phaseA-parse.txt','lap-trial-parse.txt']
results_files = ['lap_phaseA.xml','rest_phaseA.xml','lap-trial_phaseA.xml']

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

    results = semevalTask4.train_and_trial(train_file, test_file, parse_train_file, parse_test_file,
                                           use_dep=args.dep, pickled=True)
    #create results file
    f = open(test_file, 'rb')
    testd = cPickle.load(f)
    f.close()
    XMLParser.create_xml(testd['orig'], results, testd['id'], testd['idx'], out_xml_file)





