# DetOcS #

<!--- **Det**ection of **O**s**c**illations in Eclipsing Binary Light Curve**S** !--->


### ${\color\{black}\large\textsf{Det}\color{#696969}\large\textsf{ection of}\space \color{black}\large\textsf{O}\color{#696969}\large\textsf{s}\color{black}\large\textsf{c}\color{#696969}\large\textsf{illations in Eclipsing Binary Light Curve}\color{black}\large\textsf{S}}$ ###

A web implementation for (i) detecting oscillation patterns in eclipsing binary light curves and (ii) collecting new data for the training dataset with easy and fast confirmation using the iterative object detection refinement method.



### How to use ###

**1**. Clone using `git clone https://github.com/burakulas/detocs.git` 

**2**. *a*. Install required packages and libraries with `pip install -r requirements.txt`

${\space\space\space}$   *b*. Unzip EfficientDet D1 and Faster R-CNN models by consulting READ_ME files in corresponding subfolders in [models/](https://github.com/burakulas/detocs/tree/main/models).

**3**. Run the main script: `python detocs.py`

**4**. Observe the output and find something like `* Running on http://127.0.0.1:5000`

**5**. Open a web browser and enter the above address into the address bar

**6**. - For a single target:

${\space\space\space}$ *a*. Enter the name of the TESS target (*e.g. TIC48084398*).

${\space\space\space}$   *b*. Enter an orbital period or retrieve it from the [TESS EBS](https://tessebs.villanova.edu) by checking the box.

${\space\space\space}$   *c*. Enter a period factor value (*hoover over the question mark for explanation*).
   
${\space\space\space}$ - For multiple targets:

${\space\space\space}$   *a*. Click the "Browse" button and choose a file (*see [tess_ebs_sample.csv](https://github.com/burakulas/detocs/blob/main/assets/tess_ebs_sample.csv)*).



**7**. Enter the confidence threshold, a level for the probability of the presence of the object of interest in detections (*see p.780 in [Redmon et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)*).


**8**. Select a detection model (*SSD is the fastest. For Faster R-CNN and EfficientDet D1 model you may want to run the script on a GPU. YOLO modification is coming soon!*)
  
**9**. Click the "Submit" button to start the process.

**10**. *Optional*: Click the "Send to the training set" button below the resulting image to save the corresponding image and annotations in [*class, x-center, y-center, width, height*] format.

<p align="center">
   <kbd>
<img src="https://github.com/burakulas/detocs/blob/main/assets/screen2.png" alt="https://raw.githubusercontent.com/burakulas/detocs/main/assets/screen2.png?token=GHSAT0AAAAAACS2WTMNVMZDAGOTALFCLRB6ZUNCS5A" data-canonical-src="https://raw.githubusercontent.com/burakulas/detocs/main/assets/screen2.png" class="transparent shrinkToFit" width="313" height="200">
   </kbd>
&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <kbd>
<img src="https://github.com/burakulas/detocs/blob/main/assets/screen3.png" alt="https://raw.githubusercontent.com/burakulas/detocs/main/assets/screen3.png?token=GHSAT0AAAAAACS2WTMNVMZDAGOTALFCLRB6ZUNCS5A" data-canonical-src="https://raw.githubusercontent.com/burakulas/detocs/main/assets/screen3.png" class="transparent shrinkToFit" width="313" height="200">
   </kbd>
</p>





