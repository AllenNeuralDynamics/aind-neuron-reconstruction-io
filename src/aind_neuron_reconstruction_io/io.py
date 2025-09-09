"""
This module provides the io operations to support NeuronData.
"""

import io
import json
from collections import deque
from importlib.resources import files

import cloudvolume
import nrrd
import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from meshparty import meshwork
from six import iteritems

from aind_neuron_reconstruction_io.utils import (
    fix_local_cloudpath,
    get_basename,
    get_parent_dir,
    pull_mw_skel_colors,
    read_file,
)

_cached_ccf_annotation = None


def load_ccf_annotation():
    """load ccf annotation using numpy array where the axes dimensions are as follows
    0 = x = anterior-posterior
    1 = y = dorsal-ventral
    2 = z = medial-lateral

    Returns:
        numpy array: 3-d numpy array where values represent structure IDs
    """
    annotation_path = (
        files("aind_neuron_reconstruction_io")
        / "util_files/annotation_10.nrrd"
    )
    annotation, _ = nrrd.read(annotation_path)

    return annotation


def get_cached_ccf_annotation():
    """Retrieve the cached CCF annotation or load it if not already cached

    Returns:
        np.array: cached ccf annotation volume
    """
    global _cached_ccf_annotation
    if _cached_ccf_annotation is None:
        _cached_ccf_annotation = load_ccf_annotation()
    return _cached_ccf_annotation


def annotate_ccf_structure(input_df, x_col="x", y_col="y", z_col="z"):
    """given a dataframe with columns representing the x, y and z (anterior-posterior, dorsal-ventral
    and left-right, respectively), will add a column 'allenId' to the dataframe representing the
    CCF structure Id each node resides in

    Args:
        input_df (pd.DataFrame):
        x_col (str, optional): x column name. Defaults to 'x'.
        y_col (str, optional): y column name. Defaults to 'y'.
        z_col (str, optional): z column name. Defaults to 'z'.


    Returns:
        None
    """

    if not all([d in input_df.columns for d in [x_col, y_col, z_col]]):
        raise ValueError(
            "dimension column names must be in input_df columns. Columns provided: "
            f"{x_col}, {y_col}, {z_col}"
        )

    ccf_annotation = get_cached_ccf_annotation()

    def annotate_row(
        x, y, z, annotation=ccf_annotation, annotation_resolution=10
    ):
        """
        Scale a given coordinate to the atlas resolution and
        return the structure-ID for the given coordinate. Returns
        0 if the passed [x,y,z] coordinate is out of brain.

        Args:
            x (int): x-voxel
            y (int): y-voxel
            z (int): z-voxel
            annotation_resolution (int): resolution of the annotation atlas
            annotation (array): 3d numpy array with annotation data. Defaults to ccf_annotation.

        Returns:
            int: ccf structure ID
        """
        volume_shape = (1320, 800, 1140)
        voxel = [
            np.floor(x / annotation_resolution).astype(int),
            np.floor(y / annotation_resolution).astype(int),
            np.floor(z / annotation_resolution).astype(int),
        ]
        for dim in [0, 1, 2]:
            if voxel[dim] >= volume_shape[dim]:
                return 0

        return annotation[voxel[0], voxel[1], voxel[2]]

    input_df["allenId"] = input_df.apply(
        lambda row: annotate_row(x=row[x_col], y=row[y_col], z=row[z_col]),
        axis=1,
    )


def read_swc(input_file, ccf_annotate_vertices, separator=" "):
    """Will load a ccf-registered SWC file, optionally applying an offset to (x,y,z) coordinates if a line in the header has the format: # OFFSET X Y Z
    Args:
        input_file (str): Path to SWC file to load.
        ccf_annotate_vertices (bool): Whether to annotate the vertices with a
            CCF structure ID.
        separator (str): Delimiter for the SWC coordinates. Defaults to ' '.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            ['node_id', 'compartment', 'x', 'y', 'z', 'r', 'parent', 'allenId'].
    """
    # Default offset
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 0.0

    # First pass: read the file line by line to detect if an OFFSET line is present
    with open(input_file, "r") as f:
        for line in f:
            if line.strip().startswith("# OFFSET"):
                # Expected format: "# OFFSET X Y Z"
                parts = line.strip().split()
                # parts[0] = "#", parts[1] = "OFFSET", parts[2] = X, parts[3] = Y, parts[4] = Z
                if len(parts) == 5:
                    try:
                        offset_x = float(parts[2])
                        offset_y = float(parts[3])
                        offset_z = float(parts[4])
                    except ValueError:
                        raise ValueError(
                            "Invalid OFFSET line in header. "
                            "Expected numeric values after '# OFFSET': "
                            f"{line.strip()}"
                        )
                else:
                    raise ValueError(
                        "Invalid OFFSET line in SWC header. "
                        "Expected exactly 3 numbers following '# OFFSET'."
                    )
                break  # We only look for the first occurrence of OFFSET.

    # Now read the SWC file as usual, ignoring the header lines (#)
    swc_df = pd.read_csv(
        input_file,
        delim_whitespace=True,
        comment="#",
        header=None,
        names=["node_id", "compartment", "x", "y", "z", "r", "parent"],
    )

    # Apply the offset to (x, y, z)
    swc_df["x"] = swc_df["x"] + offset_x
    swc_df["y"] = swc_df["y"] + offset_y
    swc_df["z"] = swc_df["z"] + offset_z

    if ccf_annotate_vertices:
        annotate_ccf_structure(swc_df, x_col="x", y_col="y", z_col="z")
    else:
        swc_df["allenId"] = [0] * len(swc_df)

    return swc_df


def read_json(input_file, ccf_annotate_vertices):
    """
    Will load a ccf-registered mouselight json file.

    Args:
        input_file (str): path to mouselight json

    Returns:
        nrn_df: DataFrame
    """

    file_content = read_file(input_file)
    data_dict = json.loads(file_content)

    return parse_json(data_dict, ccf_annotate_vertices)


def parse_json(data_dict, ccf_annotate_vertices):
    """
    Will parse a ccf-registered mouselight json object.

    Args:
        data_dict (str): mouselight json object

    Returns:
        nrn_df: DataFrame
    """

    compartments = ["axon", "dendrite"]

    data_key = "neurons" if "neurons" in data_dict else "neuron"
    neuron_list = (
        data_dict[data_key] if data_key == "neurons" else [data_dict[data_key]]
    )

    # as per validation of the input file, there should be exactly one neuron in neuron_list
    neuron_dict = neuron_list[0]

    soma = neuron_dict["soma"]
    soma_rad = soma["radius"] if "radius" in soma else 0
    soma_struct = 0

    # check if cell has node level ccf annotations
    has_ccf_annotations = False
    if "allenId" in soma:
        soma_struct = soma["allenId"]
        has_ccf_annotations = True

    soma_data = [
        1,
        1,
        soma["x"],
        soma["y"],
        soma["z"],
        soma_rad,
        -1,
        soma_struct,
    ]

    this_neuron_node_list = [soma_data]
    node_count = 1
    for node_type_str in compartments:

        # dictionary that tracks node ids from the compartment tree level (local)
        # to their ID in the entire neuron tree (global)
        local_id_to_global_id = {1: 1}

        # get list of nodes in this compartment
        if node_type_str in neuron_dict:
            compartment_node_list = neuron_dict[node_type_str]
        else:
            compartment_node_list = []

        # look up tables for parent-child relationships
        node_id_to_node = {n["sampleNumber"]: n for n in compartment_node_list}
        parent_ids_dict = {
            nid: n["parentNumber"] for nid, n in iteritems(node_id_to_node)
        }
        child_ids_dict = {nid: [] for nid in node_id_to_node}
        for nid in parent_ids_dict:
            pid = parent_ids_dict[nid]
            if pid != -1:
                child_ids_dict[pid].append(nid)

        # find root nodes of this compartment tree
        root_nodes = [
            n for n in compartment_node_list if n["parentNumber"] == -1
        ]
        for root in root_nodes:

            if [soma["x"], soma["y"], soma["z"]] == [
                root["x"],
                root["y"],
                root["z"],
            ]:
                # this root is the same as the soma,
                # just append its children to the queue
                roots_children = [
                    node_id_to_node[i]
                    for i in child_ids_dict[root["sampleNumber"]]
                ]
                this_queue = deque(roots_children)

            else:
                # this is a root that is separate from the soma, therfore begin the
                # queue with this root
                this_queue = deque([root])

            # iterate over this tree with dfs
            while len(this_queue) > 0:

                node_count += 1
                this_node = this_queue.popleft()

                # get local (within subtree) ID and global ID (across entire neuron)
                local_id = this_node["sampleNumber"]
                global_id = node_count
                local_id_to_global_id[local_id] = global_id

                local_parent_id = this_node["parentNumber"]

                if local_parent_id == -1:
                    # this_node is a new root node
                    global_parent_id = -1
                else:
                    global_parent_id = local_id_to_global_id[local_parent_id]

                children = [
                    node_id_to_node[i] for i in child_ids_dict[local_id]
                ]
                for child in children:
                    this_queue.appendleft(child)

                node_struct = (
                    this_node["allenId"] if "allenId" in this_node else 0
                )

                record = [
                    global_id,
                    this_node["structureIdentifier"],
                    this_node["x"],
                    this_node["y"],
                    this_node["z"],
                    this_node["radius"],
                    global_parent_id,
                    node_struct,
                ]

                this_neuron_node_list.append(record)

    col_list = [
        "node_id",
        "compartment",
        "x",
        "y",
        "z",
        "r",
        "parent",
        "allenId",
    ]
    nrn_df = pd.DataFrame(this_neuron_node_list, columns=col_list)
    if not has_ccf_annotations and ccf_annotate_vertices:
        annotate_ccf_structure(nrn_df, x_col="x", y_col="y", z_col="z")

    return nrn_df


def read_h5(input_file, ccf_annotate_vertices):
    """load data from an .h5 file into a dataframe

    Args:
        input_file (str): path to .h5 file
        ccf_annotate_vertices (bool): when True will try to annotate the vertices with ccf structure ID

    Returns:
        pd.Datarame:
    """

    directory = get_parent_dir(input_file)
    filename = get_basename(input_file)
    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])
    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)
    data_df = meshwork_skeleton_to_neurondata(
        mw,
        ccf_annotate_vertices,
        get_compartment_labels=True,
        ccf_annotate_vertices=ccf_annotate_vertices,
    )

    return data_df


def read_precomputed(project_directory, skeleton_id, ccf_annotate_vertices):
    """load a precomputed neuroglancer file into a dataframe. The input project directory should
    follow the format as seen below:

    project_directory
        info
        skeleton
            1324
            1422
            1950
            info

    Where 1324, 1422 and 1950 are precomputed files. These would be valid skeleton_id values to pass
    into this function.

    Args:
        project_directory (str): path to neuroglancer project level directory
        skeleton_id (int): the integer id of the skeleton to load
        ccf_annotate_vertices (bool): if true, will try to annotate the vertices with their ccf structure label

    Returns:
        neuron_df: neuron dataframe
    """
    project_directory_cv = fix_local_cloudpath(project_directory)
    cv = cloudvolume.CloudVolume(project_directory_cv)
    skel = cv.skeleton.get(skeleton_id)

    vertices = skel.vertices
    n_vertices = len(vertices)

    edges = skel.edges

    # get compartment labels
    skel_attributes = dir(skel)
    if "compartment" in skel_attributes:
        node_types = skel.compartment
    elif "vertex_types" in skel_attributes:
        node_types = skel.vertex_types
    elif "compartments" in skel_attributes:
        node_types = skel.compartments
    else:
        node_types = np.zeros((n_vertices))

    # get allen ID
    allen_ids = np.zeros((n_vertices))
    found_allen_ids = False
    if "allenId" in skel_attributes:
        allen_ids = skel.allenId
        found_allen_ids = True

    # get presynaptic_count
    pre_syn_ct = np.zeros((n_vertices))
    if "presynaptic_count" in skel_attributes:
        pre_syn_ct = skel.presynaptic_count

    # get postsynaptic_count
    post_syn_ct = np.zeros((n_vertices))
    if "postsynaptic_count" in skel_attributes:
        post_syn_ct = skel.postsynaptic_count

    # start building neuron data
    data_list = [
        vertices,
        skel.radius.reshape(-1, 1),
        node_types.reshape(-1, 1),
        allen_ids.reshape(-1, 1),
        pre_syn_ct.reshape(-1, 1),
        post_syn_ct.reshape(-1, 1),
    ]
    data_arr = np.hstack(data_list)
    neuron_df = pd.DataFrame(
        data_arr,
        columns=[
            "x",
            "y",
            "z",
            "r",
            "compartment",
            "allenId",
            "presynaptic_count",
            "postsynaptic_count",
        ],
    )
    neuron_df["parent"] = [None] * n_vertices
    neuron_df[["x", "y", "z"]] = neuron_df[["x", "y", "z"]] / 1000

    for edge in edges:
        neuron_df.loc[edge[0], "parent"] = edge[1]

    # the root index will be the one that is not in the edge IDs
    root_index = set(np.arange(n_vertices)) - set(edges[:, 0])
    neuron_df.loc[root_index, "parent"] = -2
    neuron_df["node_id"] = neuron_df.index
    neuron_df["node_id"] = neuron_df["node_id"] + 1
    neuron_df["parent"] = neuron_df["parent"] + 1

    neuron_df = neuron_df[
        [
            "node_id",
            "compartment",
            "x",
            "y",
            "z",
            "r",
            "parent",
            "allenId",
            "presynaptic_count",
            "postsynaptic_count",
        ]
    ]

    if ccf_annotate_vertices and not found_allen_ids:
        annotate_ccf_structure(neuron_df, x_col="x", y_col="y", z_col="z")

    return neuron_df


def meshwork_skeleton_to_neurondata(
    mw,
    ccf_annotate_vertices,
    get_compartment_labels,
):
    """
    Will convert a meshwork object to a NeuronData object


    Args:
        mw (meshparty meshworks object):
        ccf_annotate_vertices (bool): when True will try to annotate the vertices with ccf annotations
        get_compartment_labels (bool): when True will add compartment labels to the skeleton

    Returns:
        NeuronData:
    """
    from aind_neuron_reconstruction_io.NeuronData import NeuronData

    vertices = mw.skeleton.vertices
    n_vertices = len(vertices)

    edges = mw.skeleton.edges
    root_index = mw.skeleton._rooted.root

    r_df = mw.anno.segment_properties.df[["r_eff", "mesh_ind_filt"]].set_index(
        "mesh_ind_filt"
    )
    radius = (
        r_df.loc[mw.skeleton_indices.to_mesh_region_point].r_eff.values / 1000
    )

    # get compartment labels
    if get_compartment_labels:
        compartment = pull_mw_skel_colors(
            mw, "basal_mesh_labels", "is_axon", "apical_mesh_labels"
        )
    else:
        compartment = np.zeros(len(radius))

    # start building neuron data
    data_list = [vertices, radius.reshape(-1, 1), compartment.reshape(-1, 1)]
    data_arr = np.hstack(data_list)

    neuron_df = pd.DataFrame(
        data_arr, columns=["x", "y", "z", "r", "compartment"]
    )
    neuron_df["parent"] = [None] * n_vertices

    for edge in edges:
        neuron_df.loc[edge[0], "parent"] = edge[1]
    neuron_df.loc[root_index, "parent"] = -2
    neuron_df["node_id"] = neuron_df.index
    neuron_df["node_id"] = neuron_df["node_id"] + 1
    neuron_df["parent"] = neuron_df["parent"] + 1

    # Quantify synapses by node
    postsyn_counts_df = pd.DataFrame(
        mw.anno.post_syn.df["post_pt_mesh_ind_filt"].value_counts()
    )
    postsyn_counts_df.columns = ["count"]
    postsyn_counts_df.index.names = ["mesh"]

    postsyn_mesh = pd.DataFrame(
        mw.mesh_indices.to_skel_index_padded, columns=["skel"]
    )
    postsyn_mesh.index.names = ["mesh"]
    postsyn_mesh["counts"] = 0
    postsyn_mesh.loc[postsyn_counts_df.index, "counts"] = postsyn_counts_df[
        "count"
    ]

    presyn_counts_df = pd.DataFrame(
        mw.anno.pre_syn.df["pre_pt_mesh_ind_filt"].value_counts()
    )
    presyn_counts_df.columns = ["count"]
    presyn_counts_df.index.names = ["mesh"]

    presyn_mesh = pd.DataFrame(
        mw.mesh_indices.to_skel_index_padded, columns=["skel"]
    )
    presyn_mesh.index.names = ["mesh"]
    presyn_mesh["counts"] = 0
    presyn_mesh.loc[presyn_counts_df.index, "counts"] = presyn_counts_df[
        "count"
    ]

    postsyn_skel_counts = (
        postsyn_mesh.groupby("skel").sum().to_dict()["counts"]
    )
    presyn_skel_counts = presyn_mesh.groupby("skel").sum().to_dict()["counts"]

    neuron_df["postsynaptic_count"] = neuron_df["node_id"].map(
        postsyn_skel_counts
    )
    neuron_df["presynaptic_count"] = neuron_df["node_id"].map(
        presyn_skel_counts
    )
    if ccf_annotate_vertices:
        annotate_ccf_structure(neuron_df, x_col="x", y_col="y", z_col="z")
    else:
        neuron_df["allenId"] = [0] * len(neuron_df)

    neuron_df = neuron_df[
        [
            "node_id",
            "compartment",
            "x",
            "y",
            "z",
            "r",
            "parent",
            "allenId",
            "presynaptic_count",
            "postsynaptic_count",
        ]
    ]

    neuron_data = NeuronData(input_data=neuron_df, ccf_annotate_vertices=False)

    return neuron_data
