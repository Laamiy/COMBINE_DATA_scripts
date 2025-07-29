from pathlib import Path
from modules.combine_log_config import  Logger 
from modules.combine_utils import const_params, handle_input , process_combine 


const                 = const_params() 
def_Output_directory  = Path().cwd().parent.resolve()/"res"/"output_data"                                #     Default output directory
default_values        = [def_Output_directory, const.num_file , const.n_output_file , const.n_VERTEX]   #     Default values for n_input_file and n_output_file
len_detec_side        = const.n_Detector_perside                                                       #     position + vibility values for one side 
#Np_files_output_dir
#-------------------------------------< Loading User inputs >-----------------------------------------------------#
try :   
    # Parsing the inputs : 
    input_args = handle_input()
    # check the validity of each input . (not None  and of the right type)
    if input_args[0] is None : 
        Logger.error("Got None as input file argument")
        exit(-1)
        
    for i in range(len(input_args)) :
        if (i == 0 or i == len(input_args )-1) : 
            continue 
        if(input_args[i] is None ) : 
            input_args[i] = default_values[ i -1 ] 
    # Check if the last argument is None, if so, assign the default value to it.
    if(input_args[-1] is None) :
      input_args[-1] = input_args[2] * const.n_sample_per_infile             # n_PHOTO_DETECTOR =  n_input_file * const.n_sample_per_infile # 5760
    # Unpack the input arguments :

except Exception as e : 
        # Show proper script usage : 
        Logger.error(f"{e}")
        Logger.info(f" Script usage is :  python3 module_name.py  -p < path_2_data > -i < num_input_file > -o < num_output_file > -n < num_photo_detector >")
        exit(-1)
    

if __name__ == '__main__':
    process_combine(input_params = input_args)
    
    
    
  
