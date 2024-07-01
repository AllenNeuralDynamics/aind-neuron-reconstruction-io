import numpy as np 
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud import storage
from urllib.parse import urlparse


def coordinates_to_voxels(coords, resolution=(10, 10, 10)):
    """ Find the voxel coordinates of spatial coordinates

    Parameters
    ----------
    coords : array
        (n, m) coordinate array. m must match the length of `resolution`
    resolution : tuple, default (10, 10, 10)
        Size of voxels in each dimension

    Returns
    -------
    voxels : array
        Integer voxel coordinates corresponding to `coords`
    """

    if len(resolution) != coords.shape[1]:
        raise ValueError(
            f"second dimension of `coords` must match length of `resolution`; "
            f"{len(resolution)} != {coords.shape[1]}")

    if not np.issubdtype(coords.dtype, np.number):
        raise ValueError(f"coords must have a numeric dtype (dtype is '{coords.dtype}')")

    voxels = np.floor(coords / resolution).astype(int)
    return voxels


def fix_local_cloudpath(cloudpath):
    """
    Ensure cloud path formatting.

    Args:
        cloudpath (str): Path to file or directory.

    Returns:
        str: Fixed cloud path.
    """
    if "://" not in cloudpath:
        dir, _ = os.path.split(cloudpath)
        if len(dir) == 0:
            cloudpath = "./" + cloudpath
        cloudpath = "file://" + cloudpath
    return cloudpath


def file_exists(path):
    """
    Check if file exists in local or cloud storage.

    Args:
        path (str): File path.

    Returns:
        bool: True if file exists, False otherwise.
    """
    if path.startswith("s3://"):
        return s3_file_exists(path)
    elif path.startswith("gs://"):
        return gcs_file_exists(path)
    else:
        return os.path.exists(path)


def s3_file_exists(path):
    """
    Check if file exists in Amazon S3.

    Args:
        path (str): S3 file path.

    Returns:
        bool: True if file exists, False otherwise.
    """
    parsed_url = urlparse(path)
    bucket_name = parsed_url.netloc
    file_key = parsed_url.path.lstrip('/')
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except NoCredentialsError:
        raise ValueError("AWS credentials not found.")
    except ClientError:
        return False


def gcs_file_exists(path):
    """
    Check if file exists in Google Cloud Storage.

    Args:
        path (str): GCS file path.

    Returns:
        bool: True if file exists, False otherwise.
    """
    parsed_url = urlparse(path)
    bucket_name = parsed_url.netloc
    file_key = parsed_url.path.lstrip('/')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_key)
    return blob.exists()


def get_parent_dir(path):
    """
    Get parent directory of a file or directory path.

    Args:
        path (str): File or directory path.

    Returns:
        str: Parent directory path.
    """
    if path.startswith("s3://") or path.startswith("gs://"):
        parsed_url = urlparse(path)
        parent_dir = '/'.join(parsed_url.path.strip('/').split('/')[:-1])
        return f"{parsed_url.scheme}://{parsed_url.netloc}/{parent_dir}"
    else:
        return os.path.dirname(path)


def get_grandparent_dir(path):
    """
    Get grandparent directory of a file or directory path.

    Args:
        path (str): File or directory path.

    Returns:
        str: Grandparent directory path.
    """
    if path.startswith("s3://") or path.startswith("gs://"):
        parsed_url = urlparse(path)
        grandparent_dir = '/'.join(parsed_url.path.strip('/').split('/')[:-2])
        return f"{parsed_url.scheme}://{parsed_url.netloc}/{grandparent_dir}"
    else:
        return os.path.dirname(os.path.dirname(path))


def get_basename(path):
    """
    Get basename of a file path.

    Args:
        path (str): File path.

    Returns:
        str: Basename of the file.
    """
    if path.startswith("s3://") or path.startswith("gs://"):
        parsed_url = urlparse(path)
        return os.path.basename(parsed_url.path)
    else:
        return os.path.basename(path)


def get_file_extension(path):
    """
    Get file extension.

    Args:
        path (str): File path.

    Returns:
        str: File extension.
    """
    if path.startswith("s3://") or path.startswith("gs://"):
        return os.path.splitext(urlparse(path).path)[1]
    else:
        return os.path.splitext(path)[1]


def create_directory(path):
    """
    Create directory in local or cloud storage.

    Args:
        path (str): Directory path.
    """
    if path.startswith("s3://"):
        s3_create_directory(path)
    elif path.startswith("gs://"):
        gcs_create_directory(path)
    else:
        os.makedirs(path, exist_ok=True)


def s3_create_directory(path):
    """
    Create directory in Amazon S3.

    Args:
        path (str): S3 directory path.
    """
    parsed_url = urlparse(path)
    bucket_name = parsed_url.netloc
    directory_key = parsed_url.path.lstrip('/') + '/'
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=bucket_name, Key=directory_key)
    except NoCredentialsError:
        raise ValueError("AWS credentials not found.")
    except ClientError as e:
        raise ValueError(f"Failed to create directory in S3: {e}")


def gcs_create_directory(path):
    """
    Create directory in Google Cloud Storage.

    Args:
        path (str): GCS directory path.
    """
    parsed_url = urlparse(path)
    bucket_name = parsed_url.netloc
    directory_key = parsed_url.path.lstrip('/') + '/'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(directory_key)
    try:
        blob.upload_from_string('')
    except Exception as e:
        raise ValueError(f"Failed to create directory in GCS: {e}")

def read_file(path_to_file):
    """
    Read a file from local filesystem, GCS, or S3.
    Args:
        path_to_file (str): The path to the file.
    Returns:
        str: The content of the file as a string.
    """
    if path_to_file.startswith("s3://"):
        return read_s3_file(path_to_file)
    elif path_to_file.startswith("gs://"):
        return read_gcs_file(path_to_file)
    else:
        return read_local_file(path_to_file)

def read_local_file(path_to_file):
    """
    Read a local file.
    Args:
        path_to_file (str): The path to the local file.
    Returns:
        str: The content of the file as a string.
    """
    with open(path_to_file, 'r') as file:
        return file.read()

def read_gcs_file(path_to_file):
    """
    Read a file from Google Cloud Storage.
    Args:
        path_to_file (str): The path to the GCS file.
    Returns:
        str: The content of the file as a string.
    """
    parsed_url = urlparse(path_to_file)
    bucket_name = parsed_url.netloc
    file_key = parsed_url.path.lstrip('/')
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_key)
    
    return blob.download_as_text()

def read_s3_file(path_to_file):
    """
    Read a file from Amazon S3.
    Args:
        path_to_file (str): The path to the S3 file.
    Returns:
        str: The content of the file as a string.
    """
    parsed_url = urlparse(path_to_file)
    bucket_name = parsed_url.netloc
    file_key = parsed_url.path.lstrip('/')
    
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        return obj['Body'].read().decode('utf-8')
    except NoCredentialsError:
        raise ValueError("AWS credentials not found.")
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"File not found: {path_to_file}")


def pull_mw_skel_colors(mw, basal_table, axon_table, apical_table):
    ''' pulls the segment properties from meshwork anno and translates into skel index
    basal node table used for general dendrite labels if no apical/basal differentiation
    apical_table is optional 
    '''
    node_labels = np.full(len(mw.skeleton.vertices), 0)
    soma_node = mw.skeleton.root
    
    basal_nodes = mw.anno[basal_table].skel_index
    node_labels[basal_nodes] = 3

    node_labels[soma_node] = 1

    axon_nodes = mw.anno[axon_table].skel_index

    if apical_table is not None:
        apical_nodes = mw.anno[apical_table].skel_index
        node_labels[apical_nodes] = 4            
    
    node_labels[axon_nodes] = 2

    if 0 in node_labels:
        print("Warning: label annotations passed give labels that are shorter than total length of skeleton nodes to label. Unassigned nodes have been labeled 0. if using pull_compartment_colors, add an option for 0 in inskel_color_map such as skel_color_map={3: 'firebrick', 4: 'salmon', 2: 'steelblue', 1: 'olive', 0:'gray'}.")

    return node_labels