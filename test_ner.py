from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

REMOVE_O = False  # Is the O mark removed during the evaluation?


def main():
    print("Read data...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    print("Load and evaluate the hmm model...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(test_word_lists,
                              word2id,
                              tag2id)
    metrics = Metrics(test_tag_lists, hmm_pred, remove_O=REMOVE_O)
    metrics.report_scores()  # Print the accuracy of each mark, recall rate, f1 score
    metrics.report_confusion_matrix()  #Print confusion matrix

    # Load and evaluate the CRF model
    print("Load and evaluate the crf model...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    # bilstm Model
    print("Load and evaluate the bilstm model...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                   bilstm_word2id, bilstm_tag2id)
    metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    print("Load and evaluate the bilstm+crf model...")
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

    ensemble_evaluate(
        [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
        test_tag_lists
    )


if __name__ == "__main__":
    main()
