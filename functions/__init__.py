import torch.optim as optim


import torch.optim as optim

def get_optimizer(config, parameters):
    # 强制把可能被字符串化的超参转换为正确类型
    print("DEBUG optim types:", type(config.optim.lr), type(config.optim.weight_decay), type(config.optim.eps),
          type(config.optim.beta1))

    lr = float(config.optim.lr)
    wd = float(config.optim.weight_decay)
    eps = float(config.optim.eps)
    beta1 = float(config.optim.beta1)
    amsgrad = bool(config.optim.amsgrad)

    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=lr, weight_decay=wd,
                          betas=(beta1, 0.999), amsgrad=amsgrad, eps=eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=lr, weight_decay=wd)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
