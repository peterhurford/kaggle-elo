import pandas as pd

from cache import load_cache, save_in_cache, is_in_cache

def get_drops(verbose=True):
    if is_in_cache('drops'):
        drops, _ = load_cache('drops', verbose=False)
        drops = list(drops['drops'].values)
    else:
        drops = []
    if verbose:
        print('There are {} drops'.format(len(drops)))
    return drops

def save_drops(drops):
    save_in_cache('drops', pd.DataFrame({'drops': list(set(drops))}), None)

def add_drops(drops, verbose=True):
    old_drops = get_drops(verbose=False)
    drops = drops + old_drops
    save_drops(drops)
    print('There are {} drops'.format(len(drops)))
    return drops
