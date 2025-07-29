# Photon Visibility Combiner

This script processes simulation data files that represent photon visibility from multiple light sources, restructures the data by separating detector readings into logical spatial regions (like top/bottom, front/back, left/right), and saves the result into multiple output files. 

## Usage

To run the script, use the following command format:

```bash
python3 combine_utils.py -i <num_input_files> -o <num_output_files> --n-vertices <num_vertices> -p <input_directory> --out-path <output_directory> -n <num_photo_detectors>
```

For example:

```bash
python3 combine_utils.py -i 58 -o 936 --n-vertices 9360 -p ./dataset --out-path ./processed -n 5760
```

## Arguments

- `-i`, `--n-input`: Number of input data files to process.
- `-o`, `--n-output`: Number of output files to create.
- `--n-vertices`: Total number of light source vertices in the simulation.
- `-p`, `--in-path`: Path to the directory containing the input data files.
- `--out-path`: Directory where the output files will be saved.
- `-n`, `--n_detector`: Total number of photo-detectors on the long side of the detector setup.

## Input Format

Each input file should contain whitespace-separated rows of numerical values in the format:

```
x y z p1 p2 p3 ... p100
```

Where `x y z` are the coordinates of a light source, and `p1` through `p100` represent photon visibility or detector hit probabilities.

Input files are expected to be named in a consistent pattern (e.g., `file_0.txt`, `file_1.txt`, ..., `file_57.txt`) and will be automatically sorted numerically.

## Output

The output will be a set of files named like `ph_0.txt`, `ph_1.txt`, etc., each containing a batch of 10 visibility entries by default. Each entry contains the light source position followed by a structured concatenation of detector readings, split and grouped based on their location (e.g., top right, bottom left, front, back).

The script ensures that the output directory exists and clears any old files before writing new ones.

## Notes

- The script will validate input paths and arguments and will terminate with a descriptive error if:
  - The input directory doesn’t exist or is empty.
  - The number of vertices or files is inconsistent with expectations.
  - Too few or too many arguments are passed.
