from src.experiment_arguments import ExperimentArguments

config_space = ExperimentArguments.read_configuration_space('config.pcs')


hps = config_space.get_hyperparameters()


for h in hps:
    if hasattr(h, 'choices'):
        print('Variable Type U')
    else:
        print('Variable Type C')

print(config_space)
