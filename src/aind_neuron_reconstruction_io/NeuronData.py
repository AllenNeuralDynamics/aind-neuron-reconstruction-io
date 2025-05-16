"""
This module provides the NeuronData class and related utilities
for validating, loading, and converting various neuron data formats.
"""

import json

import cloudvolume
import numpy as np
import pandas as pd

from aind_neuron_reconstruction_io import io
from aind_neuron_reconstruction_io.utils import (
    create_directory,
    file_exists,
    fix_local_cloudpath,
    get_basename,
    get_file_extension,
    get_grandparent_dir,
    get_parent_dir,
    read_file,
)


class NeuronData(pd.DataFrame):
    """Base class for NeuronData
    Args:

    Returns:
        NeuronData:
    """

    _metadata = [
        "path_to_file",
        "input_data",
        "ccf_annotate_vertices",
        "project_directory",
        "neuron_id",
        # '_child_ids_dict'
    ]

    @property
    def _constructor(self):
        """constructor class"""
        return NeuronData

    @property
    def _constructor_sliced(self):
        """property class"""
        return pd.Series

    def __init__(
        self,
        path_to_file: str = None,
        input_data: pd.DataFrame = None,
        ccf_annotate_vertices: bool = False,
        *args,
        **kwargs,
    ):
        """NeuronData class. Either path_to_file or input_data should be provided, not both.

        Args:
            path_to_file (str, optional): path to a neuron file can be from the following
            sources: .swc, .json (mouselight), .h5 (meshparty), precomputed (no extension).
            Defaults to None. If None is passed for path_to_file, expects input_data to be passed.

            input_data (pd.DataFrame, optional): when this is not None, will create a NeuronData
            object from this dataframe. Used when converting a meshworks skeleton to NeuronData.
            The dataframe must have the columns listed below. Defaults to None. Expected columns:
            x,y,z,radius,compartment,id,postsynaptic_count,presynaptic_count,allenId

            ccf_annotate_vertices (bool, optional): when True will annotate the NeuronData vertices
            with CCF structure ID. Defaults to False.
        """

        if (path_to_file is None) and (input_data is None):
            raise ValueError(
                "path_to_file and input_data are both None, need one or the other to create a NeuronData object"
            )
        elif (path_to_file is not None) and (input_data is not None):
            raise ValueError(
                "path_to_file and input_data are both defined, need only one to create a NeuronData object"
            )

        self.input_data = input_data
        self.path_to_file = path_to_file
        if isinstance(path_to_file, str):
            self.project_directory = (
                None  # for loading neuroglancer precomputed
            )
            self.neuron_id = None  # for loading neuroglancer precomputed
            self.ccf_annotate_vertices = ccf_annotate_vertices
            self.validate_file()
            data = self.load_data()

            # default values for non-em data
            if "presynaptic_count" not in data.columns:
                data["presynaptic_count"] = [0] * len(data)
            if "postsynaptic_count" not in data.columns:
                data["postsynaptic_count"] = [0] * len(data)

            super().__init__(data, *args, **kwargs)

        elif self.input_data is not None:
            # generate from input data frame
            self.project_directory = (
                None  # for loading neuroglancer precomputed
            )
            self.neuron_id = None  # for loading neuroglancer precomputed
            self.ccf_annotate_vertices = ccf_annotate_vertices
            super().__init__(self.input_data, *args, **kwargs)

        else:
            super().__init__(path_to_file, *args, **kwargs)

    def validate_file(self):
        "Validate input file"

        # Check file extension
        file_extension = get_file_extension(self.path_to_file)

        if file_extension not in [".swc", ".json", ".h5", ""]:
            raise ValueError(
                "File must be a .swc, .json, or an extensionless precomputed file"
            )

        # Additional checks based on content
        if file_extension == ".json":
            self.validate_json()
        elif file_extension == ".swc":
            self.validate_swc()
        elif file_extension == ".h5":
            self.validate_h5()
        else:
            self.validate_precomputed()

    def validate_json(self):
        """
        Validate a .json file containing neuron data.

        Performs the following checks:
        1. The file is valid JSON
        2. Contains exactly one 'neuron' or 'neurons' key
        3. Exactly one neuron is present
        4. Soma is present with 'x', 'y', and 'z' keys
        5. Axon/dendrite compartments (if present) contain root nodes
        """
        try:
            file_content = read_file(self.path_to_file)
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {e}")

        neuron_dicts = self._extract_neuron_list(data)
        self._validate_single_neuron(neuron_dicts)

        for neuron in neuron_dicts:
            self._validate_soma(neuron)
            self._validate_compartments(neuron)

    # ---------------------------------------------------------------------
    # Internal helper methods used only by `validate_json`
    # ---------------------------------------------------------------------
    def _extract_neuron_list(self, data):
        """
        Extract the list of neurons from the JSON data.

        Only used by `validate_json`. Ensures exactly one of 'neuron' or 'neurons' is present.
        """
        keys = list(data.keys())
        if not any(k in keys for k in ("neuron", "neurons")):
            raise ValueError("Input JSON must have 'neurons' or 'neuron' key")
        if all(k in keys for k in ("neuron", "neurons")):
            raise ValueError(
                "Input JSON contains both 'neurons' and 'neuron' keys")

        if "neurons" in data:
            return data["neurons"]
        if "neuron" in data:
            return [data["neuron"]]
        return []

    def _validate_single_neuron(self, neuron_dicts):
        """
        Ensure that exactly one neuron is present.

        Only used by `validate_json`.
        """
        if not neuron_dicts:
            raise ValueError("No neuron data found in the file")
        if len(neuron_dicts) != 1:
            raise ValueError(
                f"Each mouselight .json file should contain 1 neuron but {len(neuron_dicts)} were found"
            )

    def _validate_soma(self, neuron):
        """
        Verify that the neuron contains a valid soma node with x, y, z coordinates.

        Only used by `validate_json`.
        """
        if "soma" not in neuron:
            raise ValueError("No soma node detected in the input neuron")
        if not all(coord in neuron["soma"] for coord in ("x", "y", "z")):
            raise ValueError("'x', 'y' and 'z' are required for the soma node")

    def _validate_compartments(self, neuron):
        """
        Verify that axon and dendrite compartments (if present) contain at least one root node.

        Only used by `validate_json`.
        """
        for compartment in ("axon", "dendrite"):
            if compartment in neuron:
                nodes = neuron[compartment]
                if nodes:
                    roots = [n for n in nodes if n.get("parentNumber") == -1]
                    if not roots:
                        raise ValueError(
                            f"{compartment} compartment does not have a root node (i.e., parentNumber = -1)"
                        )

    def validate_swc(self):
        """Validate swc file"""
        try:
            file_content = read_file(self.path_to_file)
            lines = file_content.splitlines()
        except Exception as e:
            err_msg = f"Invalid SWC file: {self.path_to_file}"
            raise ValueError(err_msg) from e

    def validate_h5(self):
        """To Do"""
        # TODO
        True
        return None

    def validate_precomputed(self):
        """
        Requires:
        1) the precomputed file exists
        2) The neuroglancer files should be formatted as follows:

            project_directory
                info
                skeleton
                    1324
                    1422
                    1950
                    info
        3) that the precomputed file passed into NeuronData is an integer (e.g. 1324)
        """
        if not file_exists(self.path_to_file):
            raise ValueError(
                f"Provided input file does not exists:\n{self.path_to_file}"
            )

        # the info file should be one directory up from the provided path
        project_dir = get_grandparent_dir(self.path_to_file)
        project_info_file = f"{project_dir}/info"
        if not file_exists(project_info_file):
            raise ValueError(
                f"neuroglancer info file not found. Expected one at the following path:\n{project_info_file}"
            )
        self.project_directory = project_dir

        neuron_id = get_basename(self.path_to_file)
        try:
            neuron_id = int(neuron_id)

        except ValueError as e:
            err_msg = f"Could not interpret precomputed neuron id as integer.\nInput file:\n {self.path_to_file}\n Derived neuron id: {neuron_id}"
            raise ValueError(err_msg) from e

        self.neuron_id = neuron_id

    def load_data(self):
        """Load the data according to extension type

        Returns:
            pd.DataFrame: basis for NeuronData
        """
        # Load data based on file type
        file_extension = get_file_extension(self.path_to_file)
        if file_extension == ".json":
            neuron_df = io.read_json(
                self.path_to_file, self.ccf_annotate_vertices
            )
        elif file_extension == ".swc":
            neuron_df = io.read_swc(
                self.path_to_file, self.ccf_annotate_vertices
            )
        elif file_extension == ".h5":
            neuron_df = io.read_h5(
                self.path_to_file, self.ccf_annotate_vertices
            )
        else:
            neuron_df = io.read_precomputed(
                self.project_directory,
                self.neuron_id,
                self.ccf_annotate_vertices,
            )

        # for loop below would be usefule to add things like NeuronData.get_children(node_id)
        # parent_ids_dict = dict(zip(neuron_df["node_id"],neuron_df["parent"]))
        # child_ids_dict = { nid:[] for nid in parent_ids_dict }
        # for nid in parent_ids_dict:
        #     pid = parent_ids_dict[nid]
        #     if pid != -1:
        #         child_ids_dict[pid].append(nid)

        # create the child look up dictionary
        # self._child_ids_dict = child_ids_dict
        return neuron_df

    def to_precomputed(self, outfile):
        """
        Write NeuronData to binary precomputed file for neuroglancer. This will
        initialize a info file for the output directory, create a skeletons directory,
        add an info file there and write the skeleton to a precomputed format.

        Args:
            outfile (str): path to output file, the file name should be the "segid" i.e.
                           the unique neuroglancer id for this neuron (e.g. /path/to/1000)
                           if you want the segid to be 1000
        """
        segid = get_basename(outfile)
        output_dir = get_parent_dir(outfile)
        output_dir_cp = fix_local_cloudpath(output_dir)

        info = cloudvolume.CloudVolume.create_new_info(
            num_channels=1,
            mesh="mesh",
            layer_type="segmentation",
            data_type="uint64",  # Channel images might be 'uint8'
            # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
            encoding="raw",
            resolution=[
                1000,
                1000,
                1000,
            ],  # Voxel scaling, units are in nanometers
            voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
            skeletons="skeleton",
            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size=[512, 512, 512],  # the voxels
            volume_size=[13200, 8000, 11400],
        )
        info["segment_properties"] = "segment_properties"
        # correct location
        cv = cloudvolume.CloudVolume(
            output_dir_cp, mip=0, info=info, compress=False
        )
        cv.commit_info()

        sk_info = cv.skeleton.meta.default_info()
        sk_info["transform"] = [1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0]
        vert_atts = []

        sk_info["vertex_attributes"] = [
            {"id": "radius", "data_type": "float32", "num_components": 1},
            {"id": "compartment", "data_type": "float32", "num_components": 1},
        ]
        cv.skeleton.meta.info = sk_info
        cv.skeleton.meta.commit_info()

        vertices = self[["x", "y", "z"]].values

        node_id_toindex = dict(zip(self.node_id, self.index))
        # get edges
        edge_df = self.loc[self["parent"] != -1]
        edge_ids = edge_df.node_id.map(node_id_toindex)
        parent_ids = edge_df.parent.map(node_id_toindex)
        # relabel the edges so that they are the index aligned with the vertices
        edges_relabeled = np.column_stack((edge_ids, parent_ids))

        radius = self["r"].values.astype(np.float32)
        vertex_types = self["compartment"].values.astype(int)  # np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges_relabeled,
            radius,
            None,
            segid=segid,
            extra_attributes=sk_info["vertex_attributes"],
        )
        if "allenId" in self.columns:
            sk_cv.allenId = self.allenId.values
            vert_atts.append(
                {"id": "allenId", "data_type": "float32", "num_components": 1}
            )

        if "postsynaptic_count" in self.columns:
            sk_cv.postsynaptic_count = self.postsynaptic_count.values
            vert_atts.append(
                {
                    "id": "postsynaptic_count",
                    "data_type": "float32",
                    "num_components": 1,
                }
            )

        if "presynaptic_count" in self.columns:
            sk_cv.presynaptic_count = self.presynaptic_count.values
            vert_atts.append(
                {
                    "id": "presynaptic_count",
                    "data_type": "float32",
                    "num_components": 1,
                }
            )

        sk_cv.compartment = vertex_types

        cv.skeleton.upload(sk_cv)


def neurondata_list_to_precomputed(
    list_of_neuron_data, output_dir, neuron_ids=None
):
    """
    Will write a list of NeuronData objects to an output directory organized for neuroglancer
    visualizations.

    Args:
        list_of_neuron_data (list): list of NeuronData objects
        output_dir (str): path to output neuroglancer dir
        neuron_ids (list): list of neuron IDs (integers), if not provided will just
        use the NeuronDatas index in the list.

    """
    if not file_exists(output_dir):
        create_directory(output_dir)

    if neuron_ids is not None:
        if len(neuron_ids) != len(list_of_neuron_data):
            raise ValueError(
                f"Length of neruon_ids {len(neuron_ids)} does not match length of neuron data ({len(list_of_neuron_data)})"
            )
    else:
        neuron_ids = [i + 1 for i in range(len(list_of_neuron_data))]

    output_dir_cp = fix_local_cloudpath(output_dir)

    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        mesh="mesh",
        layer_type="segmentation",
        data_type="uint64",  # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
        encoding="raw",
        resolution=[
            1000,
            1000,
            1000,
        ],  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        skeletons="skeleton",
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[512, 512, 512],  # the voxels
        volume_size=[13200, 8000, 11400],
    )
    info["segment_properties"] = "segment_properties"
    # correct location
    cv = cloudvolume.CloudVolume(
        output_dir_cp, mip=0, info=info, compress=False
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()
    sk_info["transform"] = [1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0]
    sk_info["vertex_attributes"] = [
        {"id": "radius", "data_type": "float32", "num_components": 1},
        {"id": "allenId", "data_type": "float32", "num_components": 1},
        {"id": "compartment", "data_type": "float32", "num_components": 1},
        {
            "id": "presynaptic_count",
            "data_type": "float32",
            "num_components": 1,
        },
        {
            "id": "postsynaptic_count",
            "data_type": "float32",
            "num_components": 1,
        },
    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    skeletons = []
    for neuron_data, neuron_id in zip(list_of_neuron_data, neuron_ids):

        vertices = neuron_data[["x", "y", "z"]].values

        node_id_toindex = dict(zip(neuron_data.node_id, neuron_data.index))
        # get edges
        edge_df = neuron_data.loc[neuron_data["parent"] != -1]
        edge_ids = edge_df.node_id.map(node_id_toindex)
        parent_ids = edge_df.parent.map(node_id_toindex)
        # relabel the edges so that they are the index aligned with the vertices
        edges_relabeled = np.column_stack((edge_ids, parent_ids))

        radius = neuron_data["r"].values.astype(np.float32)
        vertex_types = neuron_data["compartment"].values.astype(
            int
        )  # np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges_relabeled,
            radius,
            None,
            segid=neuron_id,
            extra_attributes=sk_info["vertex_attributes"],
        )
        sk_cv.allenId = neuron_data.allenId.values
        sk_cv.postsynaptic_count = neuron_data.postsynaptic_count.values
        sk_cv.presynaptic_count = neuron_data.presynaptic_count.values
        sk_cv.compartment = vertex_types

        skeletons.append(sk_cv)
    cv.skeleton.upload(skeletons)
