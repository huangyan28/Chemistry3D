def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'tgcnn':
        from .network import TGCNN
        return TGCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
