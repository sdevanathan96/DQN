import os
import pickle

def soft_update(target, source, tau=0.001):
    """
    update target by target = tau * source + (1 - tau) * target
    :param target: Target network
    :param source: source network
    :param tau: 0 < tau << 1
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    update target by target = source
    :param target: Target network
    :param source: source network
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def get_class_attr(Cls) -> list:
    """
    get attribute name from Class(type)
    :param Cls:
    :return:
    """
    import re
    return [a for a, v in Cls.__dict__.items()
              if not re.match('<function.*?>', str(v))
              and not (a.startswith('__') and a.endswith('__'))]

def get_class_attr_val(cls):
    """
    get attribute name and their value from class(variable)
    :param cls:
    :return:
    """
    attr = get_class_attr(type(cls))
    attr_dict = {}
    for a in attr:
        attr_dict[a] = getattr(cls, a)
    return attr_dict


def load_obj(path):
    return pickle.load(open(path, 'rb'))