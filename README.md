# DetOcS #


  
## **${\color{red}Det\color{black}ection \space of \space \color{red}O\color{black}s\color{red}c\color{black}illations \space in \space Eclipsing \space Binary \space Light \space Curve\color{red}S}$** ##


A web implementation using the iterative object detection refinement method for (i) detecting oscillation patterns in eclipsing binary light curves and (i) collecting new data for the training dataset with easy and fast confirmation.


### How to use ###

**1**. Clone using `git clone https://github.com/burakulas/detocs.git` 

**2**. Install required packages and libraries with `pip install -r requirements.txt`

**3**. Run the main script: `python detocs.py`

**4**. Observe the output and find something like `* Running on http://127.0.0.1:5000`

**5**. Open a web browser and enter the above address into the address bar

**6**. For a single target:

${\space\space\space}$ a. Input enter the name of the TESS target (*e.g. TIC48084398*).

${\space\space\space}$   b. Enter an orbital period or retrieve it from the [TESS EBS](https://tessebs.villanova.edu) by checking the box.

${\space\space\space}$   c. Enter a period factor value (*hoover over the question mark for explanation*).

   
${\space\space\space}$ For multiple targets:

${\space\space\space}$   a. Click the "Browse" button and choose a file (*see sample_list.csv*).



**7**. Enter the confidence threshold, a level for the probability of the presence of the object of interest in detections (*see p.780 in [Redmon et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)*).


**8**. Select a detection model (*SSD is faster. For using the EfficientDet model, extract Eff_detect.zip to the main folder. You may want to run the script on a GPU when using EfficientDet D1*).
  
**9**. Click the "Submit" button to start the process.
