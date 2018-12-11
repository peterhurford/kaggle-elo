from cache import load_cache

def get_drops():
    drops, _ = load_cache('drops')
    drops = list(drops['drops'].values)
    print('There are {} drops -- {}'.format(len(drops), ', '.join(drops)))
    return drops
