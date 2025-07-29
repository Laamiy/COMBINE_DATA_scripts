import os 
import re
import sys
import csv 
import numpy as np 
from pathlib import Path
import concurrent.futures
from typing import Tuple , List
from dataclasses import dataclass
from .combine_log_config import Logger

# Group const values in one dataclass  : 
@dataclass(frozen = True)
class const_params() :
    n_sample_per_output   = 10                                        # number of light sources per output file.
    num_file              = 58                                          # Default number of input files. 
    n_VERTEX              = 9360                                         # Default number of light sources (vertex)  # 9360
    n_output_file         = n_VERTEX // n_sample_per_output               # Default number of output files # 936
    n_sample_per_infile   = 100                                            # Number of probability values of a photon hitting a detector.
    #---------------------------------------
    n_max_thread          = 4                                              # For mutithreaded outuput file writing at the end of the script.
    n_FILES               = num_file                                      # Number of files in the current dataset.
    n_Detector            =  5760                                        # Default number of photo-detectors.
    n_Detector_perside    = n_Detector // 4  # 1440                     # Number of detector on each lateral side  and with respect to  the cathode
    max_args              = 13                                         # Number of maximum argument for usage error detection purposes.
    n_detector_short_side = 1152
    Valid_data_len        = 103
#---------------------------- < List and array slices > ----------------------------#
# Combine index position every step
def list_slice(S : np.ndarray , step : int ) -> list : 
    return [S[i::step] for i in range(step)]
# Divide the list to have dim size each
def list_split(S :list  , dim : int  ) -> list : 
    return [S[i:i+dim] for i in range(0, len(S), dim)]
# Combine index position and return the sime size array, first part 
def array_slice(S :np.ndarray, step : int ) -> Tuple[np.ndarray, np.ndarray] : 
    sliced = list_slice(S,step) 
    slice2 = np.array(sliced)
    slice2 = slice2.reshape(len(S))
    odd,even = np.split(slice2,step) # Splits it into even and odd parts 
    return (odd , even)
#---------------------------- < make a chunck of 12 > ----------------------------#
def make_bunches ( detector_side : list ,reverse  = True , BUNCH_SIZE = 12) -> Tuple[np.ndarray, np.ndarray] : 
        # groupe the content of detector_side in bunches of BUNCH_SIZE (12) elements
        detector_side = [detector_side[i : i + BUNCH_SIZE] for i in range(0, len(detector_side), BUNCH_SIZE)]
        # _             =  [print(len(chunck)) for chunck in detector_side ]
        
        # Give the even / odd  positioned bunches and group them together
        b_matrix = [bunch for index, bunch in enumerate(detector_side) if (index % 2 == 0) ]
        t_matrix = [bunch for index, bunch in enumerate(detector_side) if (index % 2 != 0) ]
        # Reverse the element of  the top with respect to the bottom :
        if reverse  : 
            t_matrix = [arr[::-1] for arr in t_matrix]
        # Stack their content into numpy arrays : 
        if(not ( len(b_matrix) & len(t_matrix) ) ) :
            Logger.warning(" Empty Matrices returned ! ") 
            exit(-1)
        
        b_matrix = np.hstack(b_matrix).reshape(1,-1)
        t_matrix = np.hstack(t_matrix).reshape(1,-1)
        return (b_matrix , t_matrix)

#---------------------------- < write output files > -----------------------------#

def write_output_file(path : Path , index: int   , visibility_matrix : np.ndarray , n_sample  = const_params.n_sample_per_output) -> None :
    with open(path, 'w') as output_file:
        np.savetxt(output_file, visibility_matrix[ n_sample*index : index*n_sample + n_sample ,:], delimiter = ', ',fmt ='%1.5e')#1.3e')

def write_file(i : int,_Output_directory : Path, _Out_visibility_mat : np.ndarray):
    file_name = f'ph_{i}.txt'
    file_path = _Output_directory / file_name
    write_output_file(file_path, i, _Out_visibility_mat)

#----------------------------- < Parse input data > ------------------------------#
def get_data( data_path : str , n_files : int  , n_sample = const_params.n_sample_per_infile) -> Tuple[np.ndarray, np.ndarray] :
    dataset       = []
    files         = [f for f in os.listdir(data_path) if os.path.isfile(f)] 
    if n_files  > len(files): 
        Logger.fatal("Number of input_files exceeds the number of existing files")
        exit(-1)
    # Check if data_path contain files : exit if not
    if(not len(files)) :
        Logger.fatal("Empty directory")
        exit(-1)
    # Absolute path pointing to the dataset : 
    abs_data_path = Path(data_path).resolve()
    # Sort the files within the directory : 
    files   = sorted(files , key = lambda s : int(s.split('_')[2].split('.')[0])) # sort by the file_number by casting it into an int first.
    
    Logger.info(f"Processing {len(files)} files ...") 
    
    for index, file_name in enumerate(files):
        if index >= n_files:
            break
        with open(abs_data_path / file_name, 'r', encoding='utf-8') as f:
            for line in f:
                row = re.split(r"\s+", line.strip())
                row = [float(number) for number in row]
                dataset.append(row)
                
                
    n_vec : int   = len(dataset)
    # Separates the light source position and the visibility matrix into two matrices :
    Inputs  = np.zeros( shape = (n_vec, 3) ,  dtype  = np.float32)
    Outputs = np.zeros( shape = (n_vec, n_sample), dtype  = np.float32)
    Logger.debug(f"Dataset loaded, dataset length: {str(n_vec)}")
    
    for i in (range(n_vec)):
        event : List[int]   = dataset[i] 
        if len(event) > const_params.Valid_data_len : 
            event = event[: len(event) -1] ### !!! Temporary fix  :
        Inputs[i,:]         = event[:3]
        Outputs[i]          = event[3:]
        
    return (Inputs, Outputs)
#----------------------------- < handles user-input  > ------------------------------#

def handle_input(n_argmax = const_params.max_args)-> list :
    path_check = False
    # Number of input  files , number of output files, path to data directory : Relative / absolute.
    path_2_data , output_path, n_input_file , n_output_file, n_vertex , n_ph_detector  = (None ,None,None, None , None ,None) 
    if(len(sys.argv) > 2):
        for i, input in enumerate(sys.argv): 
            # Test input flag against each preset flags. 
            if  ( input == "-i" or input == "--n-input"):  # Number of input files
                n_input_file   = sys.argv[ i + 1 ].strip()
                n_input_file   = int(n_input_file)
            elif (input == "-o"or input == "--n-output"):  # Number of output files
                n_output_file  = sys.argv[ i + 1 ].strip()
                n_output_file  = int(n_output_file)
            elif (input == "--n-vertices" ):              # Number of vertex for the simulation 
                n_vertex       = sys.argv[ i + 1 ].strip()
                n_vertex       = int(n_vertex)                
            elif (input == "-p" or input == "--in-path" ):  # Path to the directory containing the data
                path_2_data    = sys.argv[ i + 1 ].strip()
                path_check     = Path(path_2_data).resolve().exists()
            elif (input == "--out-path"   ):              # Path to the output matrix 
                output_path    = sys.argv[ i + 1 ].strip()
            elif (input == "-n"or input == "--n_detector" ): # Number of photo_detector 
                n_ph_detector  = sys.argv[ i + 1 ].strip()
                n_ph_detector  = int(n_ph_detector)
    #Make sure the user enters the correct number of arguments before execution
    elif (len(sys.argv) < 3):
        raise Exception("Not enough input argument")
    if(len(sys.argv) > n_argmax ) :
        raise IndexError("Too much input argument")
    if(not path_check ) : 
        raise FileNotFoundError(f" No such file or directory : '{path_2_data}'")
    return [path_2_data , output_path , n_input_file , n_output_file , n_vertex , n_ph_detector]
#--------------------------------------------< Process data >---------------------------------#
def prune_files(path: Path)-> None : 
    for item in (path.iterdir()): 
        if item.is_file() : 
            item.unlink()
#---------------------------------------------------------------------------------------------#
def process_combine(input_params : list )  -> None :     
    ( input_path , output_path ,n_input_file , n_output_file , n_vertex , n_Photo_detec_long  ) =  input_params
    path_2_data_        = input_path
    num_file_           = n_input_file
    n_output            = n_output_file
    # Path pointing to the output directory in order to store the ouput files of this script : Default or not . 
    Output_directory    = Path(output_path).resolve()
    # Make sure to  create the output directory if it does not exist.
    Output_directory.mkdir(parents = True , exist_ok = True)                
    
    ph_visibility       = [] # Visibility. 
    temp                = [] # for temporary storage.
    # Get the light source position and the visibility matrix from the data files :
    light_source_pos, ph_visibility = get_data(path_2_data_, num_file_)
    light_source_pos                = light_source_pos.reshape(-1,3)
    
    # To store the output for furthe processing :
    # Out_visibility_mat  = np.zeros( shape = ( n_vertex , n_Photo_detec_long + const_params.n_detector_short_side+ 3 )) 
    Out_visibility_mat  = np.zeros( shape = ( n_vertex ,( n_Photo_detec_long // 2  )+ (const_params.n_detector_short_side //2) + 3 )) # (n_vertex , 720 + 288 + 3) 
    sample_vector       = np.zeros(const_params.n_sample_per_infile + 3)
    Logger.info("Concatenating each detector side readings ... ")
    # Remove all files in the output directory : To avoid any confusion to existing files
    prune_files(path = Output_directory)
    # DETECTOR SHORT LATERAL SIDE : 
    temp  = None
    for m in (range(n_vertex)):   
        temp =  ph_visibility[m].copy() 
        # Number of total channel in the long part 
        total_channel_vector       = temp[ 0 : n_Photo_detec_long]      # should be (1 * 5760 ) -> Meaning 1152 left

        total_detec_len = n_Photo_detec_long + const_params.n_detector_short_side
        total_channel_short_vector = temp[ n_Photo_detec_long : total_detec_len ] #!!!! should go from 5760 -> 5760 + 1152 
        total_channel_short_vector = total_channel_short_vector.reshape(1,-1)
        
        unused_part = total_detec_len - len(temp) # 152 - 40 
        total_channel_short_vector = np.concatenate((total_channel_short_vector , np.zeros((1,unused_part))) , axis = 1) #!!! 112 left unused filled with 0s ! 
        total_channel_short_vector = np.squeeze(total_channel_short_vector)
        
        ## Split into 2 right and left side
        detec_right_side , detec_left_side = array_slice(total_channel_vector,2)
        ## Split into 2 front and back side
        detec_front_side , detec_back_side = array_slice(total_channel_short_vector,2)
        ## Split into top and bottom part 
            # Front side of the detector : 
        bottom_front , top_front = make_bunches(detector_side = detec_front_side , reverse = False) # Bunch of 12 elements
            # Back side of the detector  : 
        bottom_back  , top_back  = make_bunches(detector_side = detec_back_side  , reverse = False) # Bunch of 12 elements
            # Right side of the detector : 
        bottom_right , top_right = make_bunches(detector_side = detec_right_side )   # Bunch of 12 elements
            # Left side of the detector  : 
        bottom_left , top_left   = make_bunches(detector_side = detec_left_side)     # Bunch of 12 elements
   

        # Concatenate each part in the following order :  ( top_right - bottom_right - top_left - bottom_left )
        source_pos_temp          = np.expand_dims( light_source_pos[m,:], axis = 0)
        # sample_vector            = np.concatenate((     source_pos_temp,
        #                                                 top_right, 
        #                                                 bottom_right,
        #                                                 top_left,
        #                                                 bottom_left,
        #                                                 top_front, 
        #                                                 bottom_front,
        #                                                 top_back,
        #                                                 bottom_back),
        #                                                 axis = 1
        #                                             ) # just [] + [] + [] every element. 
        sample_vector = np.concatenate([
                                        source_pos_temp,
                                        top_right[ :, n_Photo_detec_long/2: n_Photo_detec_long], # 720  until 2*720 
                                        top_front
                                        ],
                                        axis =1
                                       ) 
        sample_vector            = sample_vector.reshape(1,-1)

        # Handle incorrect value of n_ph_detector from user input : 
        try : 
            Out_visibility_mat[ m , : ] = sample_vector
        except ValueError as e : 
            Logger.fatal(f" ValueError : could not broadcast input array from shape {Out_visibility_mat[m,:].shape } into shape {sample_vector.shape}") 
            exit(-1)
        
    # ----------- < Multithreaded for writing to each file >------------------- :
    Logger.info(f"Writing output files to : {Output_directory}  ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers = const_params.n_max_thread) as executor:
    # Writes 10 rows per output file created , for ease of readability.
    # Runs each call to  write_file in pseudo-prallel for fastere execution :
        results = [ executor.submit( write_file , i, Output_directory , Out_visibility_mat ) for i in range(n_output) ]  
    # Join every thread to the main thread
        try : 
            for f in concurrent.futures.as_completed(results):                      
                f.result()
        except Exception as e : 
            Logger.error(f"Error in writing output files : {e}")
            exit(-1)
        #----------------------------------------------------------------------
    Logger.debug(f"Out_visibility_mat shape : {Out_visibility_mat.shape} ( n_vertex X (3 + n_ph_detector) )")
    # return (Out_visibility_mat , TOP_RIGHT, BOTTOM_RIGHT, TOP_LEFT, BOTTOM_LEFT ) 



    
