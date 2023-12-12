## Line Roughness for Magnesium Additive Manufacturing Samples ##
*Can also be used for other applications* 

### If you don't already have a way to run python you can install Anaconda + Spyder: ###
  1. Download anaconda from this url: https://www.anaconda.com/download/
  2. If you are on a work computer, you must open the terminal and type: conda config –-set ssl_verify <crt>
    - get the crt from IT
  3. If your anaconda navigator doesn't have spyder, type this into terminal: conda install -c anaconda spyder 

### Need to install opencv and scipy on python to use: ###
- for pip:
  - pip install opencv-python
  - python -m pip install scipy
 
- for conda:
  - conda install -c conda-forge
  - conda install scipy

### How to Use: ###
  1. Change image directory to the path with the folder of images 
  2. set show_thresh, save_thresh, and show_contours to True/False accordinly
  3. change thresh directory 
  4. set scale if it is different from the current input 
    
### Tips: ### 
- For manual thresholding, you can use ImageJ or FIJI prior to estimate threshold values:
  - From menu at the top, click Image and change "Type" to 8-bit
  - Right under "Type", click "Adjust" and then click threshold 
