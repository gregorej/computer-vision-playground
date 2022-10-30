import os
import re
from keras.utils.data_utils import get_file as keras_get_file
import boto3
import tarfile
from utils.nets import ensure_dir_exists

datasets_home = os.environ['DATASETS']

datasets_cache_dir = '/tmp/datasets'

protocol_pattern = re.compile("^[a-z]+://.*$")


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


def is_s3_path(path):
    return path.startswith('s3://')


def parse_s3_path(s3_path):
    if not is_s3_path(s3_path):
        raise ValueError('Not a valid S3 path ' + str(s3_path))
    s3_path = s3_path[len('s3://'):]
    parts = s3_path.split('/')
    bucket = parts[0]
    path = '/'.join(parts[1:])
    return bucket, path


def unpack(file_path, destination_dir):
    ensure_dir_exists(destination_dir)
    with tarfile.open(file_path) as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, destination_dir)


def load_from_s3(s3_path, local_path):
    bucket, path = parse_s3_path(s3_path)
    print("Downloading {} to {}".format(s3_path, local_path))
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(path, local_path)


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


def basename(path):
    try:
        from urllib.parse import urlparse
    except ImportError:
        from urlparse import urlparse
    from os.path import splitext, basename
    disassembled = urlparse(path)
    filename, file_ext = splitext(basename(disassembled.path))
    return filename


def download_if_needed(path):
    if is_s3_path(path):
        dataset_name = basename(path)
        ensure_dir_exists(datasets_cache_dir)
        dataset_local_path = os.path.join(datasets_cache_dir, dataset_name)
        if os.path.isdir(dataset_local_path):
            return dataset_local_path
        archive_path = os.path.join(datasets_cache_dir, dataset_name + '.tar.gz')
        load_from_s3(path + '.tar.gz', archive_path)
        unpack(archive_path, datasets_cache_dir)
        if not os.path.isdir(dataset_local_path):
            raise IOError('Downloaded the dataset but the ' + str(dataset_local_path) + ' does not exist')
        os.remove(archive_path)
        return dataset_local_path
    return path


if __name__ == '__main__':
    dataset_dir = '/home/sharky/.datasets/sosnowiecka'
    samples = load_bbox_samples(dataset_dir)
    print(len(samples))
    print(samples[0])