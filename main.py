# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main_hyperparams.py.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main_hyperparams.py.py
    FUNCTION : main
"""

import argparse
import datetime
import Config.config as configurable
from DataUtils.mainHelp import *
from DataUtils.Alphabet import *
from test import *
from trainer import Train
import random

###
from data import build_corpus
from evaluate import hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from evaluating import Metrics
from evaluate import ensemble_evaluate
###

# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)



def start_train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param config:  config
    :return:  None
    """
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()
    print("Finish Train.")


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param alphabet:  alphabet dict
    :param config:  config
    :return:  None
    """
    print("\nTesting Start......")
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = T_Inference(model=model, data=data, path_source=path_source, path_result=path_result, alphabet=alphabet,
                        use_crf=config.use_crf, config=config)
    infer.infer2file()
    print("Finished Test.")

def main():
    """
    main()
    :return:
    """
    

    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # config.add_args(key="mulu", value=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)

    # get data, iter, alphabet
    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)
    
    # get params
    get_params(config=config, alphabet=alphabet)
    
    # save dictionary
    save_dictionary(config=config)
    
    model = load_model(config)
        
          
    # print("Training Start......")
    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.test is True:
            start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
            exit()


def main_rep1(x , y):

    if x == 'train':
        # select data according to args.process
        print("读取数据...")
        train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
        dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
        test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
        ######
    
        if y == 'crf':
            crf_pred = crf_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists)
              )
            ensemble_evaluate(
            [crf_pred],
            test_tag_lists
        	)
        elif y == 'bilstm':
            bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
            lstm_pred = bilstm_train_and_eval(
                (train_word_lists, train_tag_lists),
                (dev_word_lists, dev_tag_lists),
                (test_word_lists, test_tag_lists),
                bilstm_word2id, bilstm_tag2id,
                crf=False
            )
            ensemble_evaluate(
            [lstm_pred],
            test_tag_lists
        	)
    
        elif y == 'bilstm-crf':
            crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
            # more data processing 
            train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
            train_word_lists, train_tag_lists
            )
            dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
                dev_word_lists, dev_tag_lists
            )
            test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
                test_word_lists, test_tag_lists, test=True
            )
            lstmcrf_pred = bilstm_train_and_eval(
                (train_word_lists, train_tag_lists),
                (dev_word_lists, dev_tag_lists),
                (test_word_lists, test_tag_lists),
                crf_word2id, crf_tag2id
                )
            ensemble_evaluate(
            [lstmcrf_pred],
            test_tag_lists
        	)

    else :


        HMM_MODEL_PATH = './ckpts/hmm.pkl'
        CRF_MODEL_PATH = './ckpts/crf.pkl'
        BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
        BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

        REMOVE_O = False  # 在评估的时候是否去除O标记

       
        # select data according to args.process
        print("读取数据...")
        train_word_lists, train_tag_lists, word2id, tag2id = \
            build_corpus("train")
        dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
        test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)           

        if y == 'crf':
            crf_model = load_model(CRF_MODEL_PATH)
            crf_pred = crf_model.test(test_word_lists)
            metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
            metrics.report_scores()
            metrics.report_confusion_matrix()

        elif y == 'bilstm':
            bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
            bilstm_model = load_model(BiLSTM_MODEL_PATH)
            bilstm_model.model.bilstm.flatten_parameters()  # remove warning
            lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                               bilstm_word2id, bilstm_tag2id)
            metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
            metrics.report_scores()
            metrics.report_confusion_matrix()

        elif y == 'bilstm-crf':    
            crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
            bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
            bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
            test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
                 test_word_lists, test_tag_lists, test=True
            )
            lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                   crf_word2id, crf_tag2id)
            metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
            metrics.report_scores()
            metrics.report_confusion_matrix()            
        
    exit()


def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",help="config path")
    parser.add_argument("-device", "--device", dest="device", type=str, default="cuda:0", help="device[‘cpu’,‘cuda:0’,‘cuda:1’,......]")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True, help="data process")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False, help="test model")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None, help="data[train, dev, test, None] for test model")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="predict model")
    parser.add_argument("-tr1", dest="train_repo1", type=str, help="train with repo1")
    parser.add_argument("-test1", dest="test_repo1", type=str, help="test with repo1")
    args = parser.parse_args()
    # print(vars(args))
    config = configurable.Configurable(config_file=args.config_file)
    config.device = args.device
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    # config
    
    if args.train_repo1 == 'crf' :
        main_rep1('train' , 'crf')
    elif args.train_repo1 == 'bilstm':
        main_rep1('train' , 'bilstm')  
    elif args.train_repo1 =='bilstm-crf':
        main_rep1('train' , 'bilstm-crf')
    elif args.test_repo1 == 'crf' :
        main_rep1('test' , 'crf')
    elif args.test_repo1 == 'bilstm':
        main_rep1('test' , 'bilstm')  
    elif args.test_repo1 =='bilstm-crf':
        main_rep1('test' , 'bilstm-crf')          
    else:
    
        if config.test is True:
            config.train = False
        if config.t_data not in [None, "train", "dev", "test"]:
            print("\nUsage")
            parser.print_help()
            print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
            exit()
        print("***************************************")
        print("Device : {}".format(config.device))
        print("Data Process : {}".format(config.process))
        print("Train model : {}".format(config.train))
        print("Test model : {}".format(config.test))
        print("t_model : {}".format(config.t_model))
        print("t_data : {}".format(config.t_data))
        print("predict : {}".format(config.predict))
        print("***************************************")

        return config


if __name__ == "__main__":

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    config = parse_argument()

    if config.device != cpu_device:
        print("Using GPU To Train......")
        device_number = config.device[-1]
        torch.cuda.set_device(int(device_number))
        print("Current Cuda Device {}".format(torch.cuda.current_device()))
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    
    main()

