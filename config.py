# _*_ coding:utf-8 _*_


class Config:
    random_seed = 9185
    np_random_seed = 3145
    context_size = 6
    vectors_dim = 50 # word vectors dimention
    ANNEAL_EVERY = 20000 # Anneal learning rate every several iterations
    SAVE_PARAMS_EVERY = 5000
    expcost = None
    step = 0.3 # learn rate
    iterations = 100000 # iterations number
    PRINT_EVERY = 10
    batchsize = 50
    tablesize = 10000 # negative samples table size
    is_save = False # save action

    def __init__(self, args):
        pass
