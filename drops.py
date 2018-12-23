import pandas as pd

from cache import load_cache, save_in_cache

def get_drops():
    if is_in_cache('drops'):
        drops, _ = load_cache('drops')
        drops = list(drops['drops'].values)
    else:
        drops = []
    print('There are {} drops'.format(len(drops)))
    return drops

def save_drops(drops):
	save_in_cache('drops', pd.DataFrame({'drops': list(set(drops))}), None)

def add_drops(drops):
	old_drops = get_drops()
	save_drops(drops + old_drops)
