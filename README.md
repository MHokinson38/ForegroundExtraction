# Foreground Extraction
Matthew Hokinson, matthew@hokinson.com

Various foreground object extraction methods 

## GraphCut

Implementation of foreground object extraction method described in [Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images](https://www.csd.uwo.ca/~yboykov/Papers/iccv01.pdf) 

### Description 

GraphCut is a minimum graph-cut based smart scissors implementation for foreground object extraction in an image. This foreground extraction works with a set of hard boundaries (seeds set by the user on the image foreground/background), as well as soft boundaries which are created by the program using two types of weights between image pixels, regional and boundary. The former is set based on the likelihood that a pixel belongs to the foreground or background based on the seeded pixels (i.e. a distribution of known intensity distributions for a pixel's foreground or background). The latter is set purely on the intensity relationship between neighboring pixels (this could be 4-neighbor or 8-neighbor). 

### Results/Demo

To come later.

## Usage 

To run the code, ensure all dependencies (listed below) are included, then open up the folder and run `foregroundExtraction.py`. This can (and should) be run with at least one argument, the image to cut (example images are included in the `ExampleImages` directory), like so `python foregroundExtraction.py <image>`. Additional arguments can be found with `python foregroundExtraction.py -h`. 

For setting the seeds, hover over the image and left click for foreground seeds, or right click for background seeds. You can run the extraction with 'Enter', and quit by pressing 'q'. 

## Dependencies 
- `numpy version=1.23.5` 
- `opencv-contrib-python version=4.6.0.66`
- `maxflow version=0.0.1`

## Tests 
You can run unit tests (located in `tests/`) by running the command `python3 -m pytest` in the root directory of the repository. 

## Coming in the future 
Currently only implementing GraphCut, but with hopes to add various other Foreground Image extraction methods for comparison in the future. 

Last updated: 12/6/22
