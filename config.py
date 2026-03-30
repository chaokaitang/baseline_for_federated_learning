# GLOBAL PARAMETERS
DATASETS = ['mnist', 'emnist']
TRAINERS = {
    'fedavg': 'FedAvgTrainer',
    'fedprox': 'FedProxTrainer',
    'ditto': 'DittoTrainer',
    'pfedme': 'PFedMeTrainer',
    'da_fedcl': 'DAFedCLTrainer'
}
OPTIMIZERS = TRAINERS.keys()

class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist' :
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 10}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 10}

        if dataset == 'emnist':
            # EMNIST-balanced: 47 classes, images are 28x28 grayscale
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 47}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 47}

        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()