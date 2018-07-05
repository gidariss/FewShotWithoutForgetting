config = {}
# set the parameters related to the training and testing set

nKbase = 389 

data_train_opt = {}
data_train_opt['nKnovel'] = 0
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = 0
data_train_opt['nTestNovel'] = 0
data_train_opt['nTestBase'] = 400
data_train_opt['batch_size'] = 1
data_train_opt['epoch_size'] = 4000
config['data_train_opt'] = data_train_opt

config['max_num_epochs'] = 100  

networks = {}
net_optim_paramsF = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]}
networks['feat_model'] = {'def_file': 'architectures/ResNetFeat.py', 'pretrained': None, 'opt': {'userelu': False, 'restype': 'ResNet10'},  'optim_params': net_optim_paramsF}

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]}
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'none', 'nKall': 1000, 'nFeat':512, 'scale_cls': 10}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': None, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
