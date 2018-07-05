config = {}
# set the parameters related to the training and testing set

nKbase = 64 
nKnovel = 5
nExemplars = 5

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = -1
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel * 3
data_train_opt['nTestBase'] = nKnovel * 3 
data_train_opt['batch_size'] = 8
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000

data_test_opt = {}
data_test_opt['nKnovel'] = nKnovel
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = nExemplars
data_test_opt['nTestNovel'] = 15 * data_test_opt['nKnovel']
data_test_opt['nTestBase'] = 15 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 2000

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': False, 'in_planes':3, 'out_planes':[64,64,64,64], 'num_stages':4}
pretrainedF = './experiments/miniImageNet_Conv64CosineClassifier/feat_model_net_epoch*.best'
networks['feat_model'] = {'def_file': 'architectures/ConvNet.py', 'pretrained': pretrainedF, 'opt': net_optionsF, 'optim_params': None}

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(10*2, 0.1),(20*2, 0.006),(25*2, 0.0012),(30*2, 0.00024)]}
pretrainedC = './experiments/miniImageNet_Conv64CosineClassifier/classifier_net_epoch*.best'
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'attention_based', 'nKall': nKbase, 'nFeat':64*5*5, 'scale_cls': 10, 'scale_att': 10.0}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': pretrainedC, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'FewShot'
