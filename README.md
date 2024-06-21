# DetOcS #


  
## **${\color{red}Det\color{black}ection \space of \space \color{red}O\color{black}s\color{red}c\color{black}illation \space in \space Eclipsing \space Binary \space Light \space Curve\color{red}S}$** ##


 A web implementation for detecting oscillation patterns in eclipsing binary light curves and quickly increasing the number of training data using the iterative object detection refinement method.

### How to use ###

**1**. Clone using `git clone https://github.com/burakulas/detocs.git` 

**2**. Install required packages and libraries with `pip install -r requirements.txt`

**3**. Run the main script: `python detocs.py`

**4**. Observe the output and find something like `* Running on http://127.0.0.1:5000`

**5**. Open a web browser and enter the above address into the address bar

**6**. For a single target:

${\space\space\space}$ a. Input enter the name of the TESS target (*e.g. TIC48084398*).

${\space\space\space}$   b. Enter an orbital period or retrieve it from the catalog by clicking the box.

${\space\space\space}$   c. Enter a period factor value (*hoover on the question mark for explanation.*)

${\space\space\space}$   d. Continue to step 7.
   
${\space\space\space}$ For multiple targets:

${\space\space\space}$   a. Click "Browse" button and choose a file (*see sample_list.csv*)

${\space\space\space}$   b. Continue to step 7.


**7**. Enter the confidence level, the probability of the presence of the object of interest in detections (*see p.788 in [Redmon et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)*)


**8**. Select a detection model (*SSD is faster. For using the EfficientDet model, extract Eff_detect.zip to the main folder. You may want to run the script on a GPU when using EfficientDet D1*)
  
**9**. Click "Search" to start the process.
