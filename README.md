# DetOcS #

<!--- **Det**ection of **O**s**c**illations in Eclipsing Binary Light Curve**S** !--->


### ${\color\{black}\large\textsf{Det}\color{#696969}\large\textsf{ection of}\space \color{black}\large\textsf{O}\color{#696969}\large\textsf{s}\color{black}\large\textsf{c}\color{#696969}\large\textsf{illations in Eclipsing Binary Light Curve}\color{black}\large\textsf{S}}$ ###

A web implementation using the iterative object detection refinement method for (i) detecting oscillation patterns in eclipsing binary light curves and (i) collecting new data for the training dataset with easy and fast confirmation.



### How to use ###

**1**. Clone using `git clone https://github.com/burakulas/detocs.git` 

**2**. Install required packages and libraries with `pip install -r requirements.txt`

**3**. Run the main script: `python detocs.py`

**4**. Observe the output and find something like `* Running on http://127.0.0.1:5000`

**5**. Open a web browser and enter the above address into the address bar

**6**. - For a single target:

${\space\space\space}$ *a*. Input enter the name of the TESS target (*e.g. TIC48084398*).

${\space\space\space}$   *b*. Enter an orbital period or retrieve it from the [TESS EBS](https://tessebs.villanova.edu) by checking the box.

${\space\space\space}$   *c*. Enter a period factor value (*hoover over the question mark for explanation*).

   
${\space\space\space}$ - For multiple targets:

${\space\space\space}$   *a*. Click the "Browse" button and choose a file (*see sample_list.csv*).



**7**. Enter the confidence threshold, a level for the probability of the presence of the object of interest in detections (*see p.780 in [Redmon et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)*).


**8**. Select a detection model (*SSD is the fastest. For [EfficientDet D1](https://github.com/burakulas/detocs/tree/main/effdet) model you may want to run the script on a GPU. and [Faster R-CNN](https://github.com/burakulas/detocs/tree/main/frcnn) modification coming soon!
  
**9**. Click the "Submit" button to start the process.

**10**. *Optional*: Click the "Send to the training set" button to save the corresponding image and annotations in [*class, x-center, y-center, width, height*] format.

## ##
### Detection on Kepler EBS ###

***detocs_k.py*** is a modified version of *detocs.py*. It was written to test the known systems with *Kepler* data and to detect patterns on [Kepler EBS](https://archive.stsci.edu/kepler/eclipsing_binaries.html) dataset. The code detects patterns on short cadence data of given systems having a KIC number. [*kepler_ebs_sample.csv*](https://github.com/burakulas/detocs/blob/main/kepler_ebs_sample.csv) is a sample that the code can read. It creates a folder with the timestamp in the name and moves the images with annotations there. A summary file is also created to check the confidence values. Run by `python detocs_k.py`
