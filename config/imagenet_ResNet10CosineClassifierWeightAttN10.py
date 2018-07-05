config = {}
# set the parameters related to the training and testing set

nKbase = 389 
nKnovel = 250
nExemplars = 10

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel * 6
data_train_opt['nTestBase'] = nKnovel * 3 
data_train_opt['batch_size'] = 1
data_train_opt['epoch_size'] = 4000

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 6

networks = {}

networks['feat_model'] = {'def_file': 'architectures/DumbFeat.py', 'pretrained': None, 'opt': {'dropout': 0.5},  'optim_params': None }

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(4, 0.01), (6, 0.001)]}
pretrainedC = './experiments/imagenet_ResNet10CosineClassifier/classifier_net_epoch100'
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'attention_based', 'nKall': 1000, 'nFeat': 512, 'scale_cls': 10, 'scale_att': 30.0}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': pretrainedC, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}
config['networks'] = networks

criterions = {}
criterions['loss']    = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['data_dir'] = './data/IMAGENET/imagenet_ResNet10CosineClassifier'
