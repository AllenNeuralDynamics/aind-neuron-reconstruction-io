import json
import numpy as np
import cloudvolume
import pandas as pd
from aind_neuron_reconstruction_io import io
from aind_neuron_reconstruction_io.utils import (fix_local_cloudpath, file_exists,
                                                     get_parent_dir, get_grandparent_dir, 
                                                     get_basename, get_file_extension, 
                                                     create_directory, read_file,
                                                     pull_mw_skel_colors)

class NeuronData(pd.DataFrame):
    
    _metadata = [
        'path_to_file', 
        'input_data', 
        'ccf_annotate_vertices', 
        'project_directory', 
        'neuron_id', 
        '_child_ids_dict'
        ]
    
    @property
    def _constructor(self):
        return NeuronData

    @property
    def _constructor_sliced(self):
        return pd.Series  
    def __init__(self, 
                 path_to_file: str = None, 
                 input_data: pd.DataFrame = None, 
                 ccf_annotate_vertices: bool = False, 
                 *args, **kwargs
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
            raise ValueError("path_to_file and input_data are both None, need one or the other to create a NeuronData object")
        elif (path_to_file is not None) and (input_data is not None):
            raise ValueError("path_to_file and input_data are both defined, need only one to create a NeuronData object")
        
        self.input_data = input_data
        self.path_to_file = path_to_file
        if isinstance(path_to_file, str):
            self.project_directory = None # for loading neuroglancer precomputed
            self.neuron_id = None # for loading neuroglancer precomputed
            self.ccf_annotate_vertices = ccf_annotate_vertices
            self.validate_file()
            data = self.load_data()
            
            # default values for non-em data
            if "presynaptic_count" not in data.columns:
                data['presynaptic_count'] = [0]*len(data)
            if "postsynaptic_count" not in data.columns:
                data['postsynaptic_count'] = [0]*len(data)
            
            super().__init__(data, *args, **kwargs)
            
        elif self.input_data is not None:
            # generate from input data frame 
            self.project_directory = None # for loading neuroglancer precomputed
            self.neuron_id = None # for loading neuroglancer precomputed
            self.ccf_annotate_vertices = ccf_annotate_vertices
            super().__init__(self.input_data, *args, **kwargs)
            
        else:
            super().__init__(path_to_file, *args, **kwargs)
        

    def validate_file(self):
        "Validate input file"
        
        # Check file extension
        file_extension = get_file_extension(self.path_to_file)
        
        if file_extension not in ['.swc', '.json', '.h5', '']:
            raise ValueError("File must be a .swc, .json, or an extensionless precomputed file")
        
        # Additional checks based on content
        if file_extension == '.json':
            self.validate_json()
        elif file_extension == '.swc':
            self.validate_swc()
        elif file_extension == ".h5":
            self.validate_h5()
        else:
            self.validate_precomputed()


    def validate_json(self):
        """
        Will verify:
        1) the data can be loaded with json
        2) there exists neuron data
        1) the soma is present and it has all required data
        2) if there are axon or dendrite nodes present that they have at least one root node
            pointing it back to the soma.
        """

        try:
            file_content = read_file(self.path_to_file)
            data = json.loads(file_content)
            # Add your JSON validation logic here using json_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {e}")

        if "neurons" not in data:
            raise ValueError("input json must have 'neurons' key")
        
        neuron_dicts = data['neurons']
        if neuron_dicts == []:
            raise ValueError("at least one neuron must be contained within the 'neurons' value")

        if len(neuron_dicts) != 1:
            raise ValueError(f"Each mouselight .json file should contain 1 neuron but {len(neuron_dicts)} were found")

    
        for neuron_dictionary in neuron_dicts:  
            if "soma" not in neuron_dictionary:
                raise ValueError("No soma node detected in the input neuron ") 
            if not all([key in neuron_dictionary['soma'] for key in ['x','y','z']]):
                raise ValueError("'x', 'y' and 'z' are required for the soma node")
            
            for compartment in ['axon', 'dendrite']:

                if compartment in neuron_dictionary:
                    these_nodes = neuron_dictionary[compartment]
                    if these_nodes:
                        roots = [n for n in these_nodes if n['parentNumber']==-1]
                        if roots == []:
                            raise ValueError(f"{compartment} compartment does not have a root node (i.e. a node with parent = -1) ") 
              
                            
    def validate_swc(self):
        file_content = read_file(self.path_to_file)
        lines = file_content.splitlines()

        try:
            file_content = read_file(self.path_to_file)
            lines = file_content.splitlines()
        except Exception as e:
            err_msg = f"Invalid SWC file: {self.path_to_file}"
            raise ValueError(err_msg) from e

    def validate_h5(self):
        #TODO
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
            raise ValueError(f"Provided input file does not exists:\n{self.path_to_file}")

        # the info file should be one directory up from the provided path
        project_dir = get_grandparent_dir(self.path_to_file)
        project_info_file = f"{project_dir}/info"
        if not file_exists(project_info_file):
            raise ValueError(f"neuroglancer info file not found. Expected one at the following path:\n{project_info_file}")        
        self.project_directory = project_dir
        
        neuron_id = get_basename(self.path_to_file)
        try:
            neuron_id = int(neuron_id)
            
        except ValueError as e:
            err_msg = f"Could not interpret precomputed neuron id as integer.\nInput file:\n {self.path_to_file}\n Derived neuron id: {neuron_id}"
            raise ValueError(err_msg) from e
        
        
        self.neuron_id = neuron_id
        
        
    def load_data(self):
        
        # Load data based on file type
        file_extension = get_file_extension(self.path_to_file)
        if file_extension == '.json':
            neuron_df =  io.read_json(self.path_to_file, self.ccf_annotate_vertices)
        elif file_extension == '.swc':
            neuron_df = io.read_swc(self.path_to_file, self.ccf_annotate_vertices)
        elif file_extension == ".h5":
            neuron_df  = io.read_h5(self.path_to_file, self.ccf_annotate_vertices)
        else:
            neuron_df = io.read_precomputed(self.project_directory, self.neuron_id, self.ccf_annotate_vertices)

        # for loop below is useful if we want to add things like NeuronData.get_children(node_id)
        parent_ids_dict = dict(zip(neuron_df["node_id"],neuron_df["parent"]))
        child_ids_dict = { nid:[] for nid in parent_ids_dict }
        # for nid in parent_ids_dict:
        #     pid = parent_ids_dict[nid]
        #     if pid != -1:
        #         child_ids_dict[pid].append(nid)
        
        # create the child look up dictionary
        self._child_ids_dict = child_ids_dict
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
            num_channels    = 1,
            mesh            = "mesh",
            layer_type      = 'segmentation',
            data_type       = 'uint64', # Channel images might be 'uint8'
            # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
            encoding        = 'raw', 
            resolution      = [1000,1000,1000], # Voxel scaling, units are in nanometers
            voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
            skeletons        = 'skeleton',
            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size      = [ 512,512,512], #the voxels
            volume_size = [13200,8000,11400]
        )
        info['segment_properties']='segment_properties'
        #correct location
        cv = cloudvolume.CloudVolume(output_dir_cp , mip=0, info=info, compress=False)
        cv.commit_info()
        
        sk_info = cv.skeleton.meta.default_info()
        sk_info['transform'] = [1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0]
        sk_info['vertex_attributes'] = [
            { 'id': 'radius',
                'data_type': 'float32',
                'num_components': 1
            },
            {
                'id': 'allenId',
                'data_type': 'float32',
                'num_components': 1
            },
            {
                'id': 'compartment',
                'data_type': 'int',
                'num_components': 1
            },
            {
                'id': 'presynaptic_count',
                'data_type': 'float32',
                'num_components': 1
            },
            {
                'id': 'postsynaptic_count',
                'data_type': 'float32',
                'num_components': 1
            }

        ]
        cv.skeleton.meta.info = sk_info
        cv.skeleton.meta.commit_info()

        vertices = self[['x','y','z']].values

        node_id_toindex = dict(zip(self.node_id, self.index))
        # get edges
        edge_df = self.loc[self['parent']!=-1]
        edge_ids = edge_df.node_id.map(node_id_toindex)
        parent_ids = edge_df.parent.map(node_id_toindex)
        # relabel the edges so that they are the index aligned with the vertices
        edges_relabeled = np.column_stack((edge_ids, parent_ids))

        radius = self['r'].values.astype(np.float32)
        vertex_types = self['compartment'].values.astype(int) #np.float32)

        sk_cv = cloudvolume.Skeleton(vertices,
                                     edges_relabeled,
                                     radius,
                                     None, 
                                     segid=segid,
                                     extra_attributes = sk_info['vertex_attributes']
                                     )
        sk_cv.allenId = self.allenId.values
        sk_cv.postsynaptic_count = self.postsynaptic_count.values
        sk_cv.presynaptic_count = self.presynaptic_count.values
        sk_cv.compartment = vertex_types
        
        cv.skeleton.upload(sk_cv)
        
        
        
def neurondata_list_to_precomputed(list_of_neuron_data, output_dir, neuron_ids = None):
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
        if len(neuron_ids)!=len(list_of_neuron_data):
            raise ValueError(f"Length of neruon_ids {len(neuron_ids)} does not match length of neuron data ({len(list_of_neuron_data)})")
    else:
        neuron_ids = [i+1 for i in range(len(list_of_neuron_data))]
        
    output_dir_cp = fix_local_cloudpath(output_dir)

    info = cloudvolume.CloudVolume.create_new_info(
        num_channels    = 1,
        mesh            = "mesh",
        layer_type      = 'segmentation',
        data_type       = 'uint64', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
        encoding        = 'raw', 
        resolution      = [1000,1000,1000], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        skeletons        = 'skeleton',
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 512,512,512], #the voxels
        volume_size = [13200,8000,11400]
    )
    info['segment_properties']='segment_properties'
    #correct location
    cv = cloudvolume.CloudVolume(output_dir_cp , mip=0, info=info, compress=False)
    cv.commit_info()
    
    sk_info = cv.skeleton.meta.default_info()
    sk_info['transform'] = [1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0]
    sk_info['vertex_attributes'] = [
        { 'id': 'radius',
            'data_type': 'float32',
            'num_components': 1
        },
        {
            'id': 'allenId',
            'data_type': 'float32',
            'num_components': 1
        },
        {
            'id': 'compartment',
            'data_type': 'int',
            'num_components': 1
        },
        {
            'id': 'presynaptic_count',
            'data_type': 'float32',
            'num_components': 1
        },
        {
            'id': 'postsynaptic_count',
            'data_type': 'float32',
            'num_components': 1
        }

    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    skeletons = []
    for neuron_data,neuron_id in zip(list_of_neuron_data,neuron_ids):
            
        vertices = neuron_data[['x','y','z']].values

        node_id_toindex = dict(zip(neuron_data.node_id, neuron_data.index))
        # get edges
        edge_df = neuron_data.loc[neuron_data['parent']!=-1]
        edge_ids = edge_df.node_id.map(node_id_toindex)
        parent_ids = edge_df.parent.map(node_id_toindex)
        # relabel the edges so that they are the index aligned with the vertices
        edges_relabeled = np.column_stack((edge_ids, parent_ids))

        radius = neuron_data['r'].values.astype(np.float32)
        vertex_types = neuron_data['compartment'].values.astype(int) #np.float32)
        
        sk_cv = cloudvolume.Skeleton(vertices,
                                        edges_relabeled,
                                        radius,
                                        None, 
                                        segid=neuron_id,
                                        extra_attributes = sk_info['vertex_attributes']
                                        )
        sk_cv.allenId = neuron_data.allenId.values
        sk_cv.postsynaptic_count = neuron_data.postsynaptic_count.values
        sk_cv.presynaptic_count = neuron_data.presynaptic_count.values
        sk_cv.compartment = vertex_types
        
        skeletons.append(sk_cv)
    cv.skeleton.upload(skeletons)
    
    
def meshwork_skeleton_to_neurondata(mw, ccf_annotate_vertices, get_compartment_labels):
    """
    Will convert a meshwork object to a NeuronData object


    Args:
        mw (meshparty meshworks object): 
        ccf_annotate_vertices (bool): when True will try to annotate the vertices with ccf annotations 
        get_compartment_labels (bool): when True will add compartment labels to the skeleton

    Returns:
        NeuronData: 
    """
        
    vertices = mw.skeleton.vertices
    n_vertices = len(vertices)

    edges = mw.skeleton.edges
    root_index = mw.skeleton._rooted.root

    r_df = mw.anno.segment_properties.df[['r_eff', 'mesh_ind_filt']].set_index('mesh_ind_filt')
    radius = r_df.loc[mw.skeleton_indices.to_mesh_region_point].r_eff.values/1000

    # get compartment labels
    if get_compartment_labels:    
        compartment = pull_mw_skel_colors(mw, 'basal_mesh_labels', 'is_axon', 'apical_mesh_labels')
    else:
        compartment =  np.zeros(len(radius))
        
    # start building neuron data 
    data_list = [
        vertices,
        radius.reshape(-1,1),
        compartment.reshape(-1,1)
    ]
    data_arr =  np.hstack(data_list)

    neuron_df = pd.DataFrame(data_arr, columns = ['x','y','z','r','compartment'])
    neuron_df['parent'] = [None]*n_vertices

    for edge in edges:
        neuron_df.loc[edge[0],'parent'] = edge[1]
    neuron_df.loc[root_index, 'parent'] = -2
    neuron_df['node_id'] = neuron_df.index
    neuron_df['node_id'] = neuron_df['node_id']+1
    neuron_df['parent'] = neuron_df['parent']+1

    # Quantify synapses by node
    postsyn_counts_df = pd.DataFrame(mw.anno.post_syn.df['post_pt_mesh_ind_filt'].value_counts())
    postsyn_counts_df.columns=['count']
    postsyn_counts_df.index.names = ['mesh']

    postsyn_mesh = pd.DataFrame(mw.mesh_indices.to_skel_index_padded, columns=['skel'])
    postsyn_mesh.index.names=['mesh']
    postsyn_mesh['counts'] = 0
    postsyn_mesh.loc[postsyn_counts_df.index, 'counts'] = postsyn_counts_df['count']

    presyn_counts_df = pd.DataFrame(mw.anno.pre_syn.df['pre_pt_mesh_ind_filt'].value_counts())
    presyn_counts_df.columns=['count']
    presyn_counts_df.index.names = ['mesh']

    presyn_mesh = pd.DataFrame(mw.mesh_indices.to_skel_index_padded, columns=['skel'])
    presyn_mesh.index.names=['mesh']
    presyn_mesh['counts'] = 0
    presyn_mesh.loc[presyn_counts_df.index, 'counts'] = presyn_counts_df['count']

    postsyn_skel_counts = postsyn_mesh.groupby('skel').sum().to_dict()['counts']
    presyn_skel_counts = presyn_mesh.groupby('skel').sum().to_dict()['counts']  


    neuron_df['postsynaptic_count'] = neuron_df['node_id'].map(postsyn_skel_counts)
    neuron_df['presynaptic_count'] = neuron_df['node_id'].map(presyn_skel_counts)
    if ccf_annotate_vertices:    
        io.annotate_ccf_structure(neuron_df, x_col='x', y_col='y', z_col='z')
    else:
        neuron_df['allenId']=[0]*len(neuron_df)
        
    
    neuron_df = neuron_df[['node_id','compartment','x','y','z','r','parent','allenId','presynaptic_count','postsynaptic_count']]
        
    neuron_data = NeuronData(input_data=neuron_df,ccf_annotate_vertices=False)
    
    return neuron_data

