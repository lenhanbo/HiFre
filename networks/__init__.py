def get_model(name):
    # Keep package imports lightweight: frequency.py imports networks.module,
    # while the trainer imports frequency.py.
    if name != 'adversarial_model':
        raise KeyError(name)
    from networks.model import AdversarialModel
    return AdversarialModel
