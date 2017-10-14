import os
import re

from keras.utils.data_utils import get_file as keras_get_file

datasets_home = os.environ['DATASETS']

datasets_cache_dir = '/tmp/datasets'

protocol_pattern = re.compile("^(http|https)://.*$")


def load_bbox_samples(dataset_dir, separator=' ', categories=['truck', 'car']):
    """
    Currently loads only car or truck bboxes
    :param separator: separator to be used when reading lables CSV file. Space is default
    :param dataset_dir: directory containing the dataset
    :return: list of tuples, where first element is file name and second is list of bbox coordinates (xmin, ymin, xmax, ymax)
    """
    import pandas as pd
    labels_file_path = get_file_through_cache('labels.csv', dataset_dir)
    df = pd.read_csv(labels_file_path, header=0, delimiter=separator)
    df = df[df['Label'].str.lower().isin(categories)].reset_index()
    samples = df.groupby(df.Frame)[['xmin', 'ymin', 'xmax', 'ymax']].apply(lambda x: [tuple(y) for y in x.values]).to_dict().items()
    samples = map(lambda x: (os.path.join(dataset_dir, x[0]), x[1]), samples)
    return sorted(samples, key=lambda sample: sample[0])


def get_file_through_cache(fname, directory=None):
    if directory is None:
        parts = fname.split('/')
        fname = parts[-1]
        directory = '/'.join(parts[0:-1])
    is_remote = protocol_pattern.match(directory)
    if is_remote:
        return keras_get_file(fname, directory)
    else:
        return os.path.join(directory, fname)


if __name__ == '__main__':
    dataset_dir = '/home/sharky/.datasets/sosnowiecka'
    samples = load_bbox_samples(dataset_dir)
    print len(samples)
    print samples[0]