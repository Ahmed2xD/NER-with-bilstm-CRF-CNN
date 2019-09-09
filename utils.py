import pickle


def merge_maps(dict1, dict2):
    """ is used to merge two word2ids or two tag2ids """
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """ is used to save the model """
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model_1(file_name):
    """ is used to load the model """
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM model training needs to add PAD and UNK in word2id and tag2id
# If it is a lstm with CRF, add <start> and <end> (required for decoding)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # If you added bilstm with CRF, then add <start> and <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # If it is test data, you don't need to add end token.
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
