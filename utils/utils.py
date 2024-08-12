import pandas as pd
import numpy as np
import os
import sys
import shutil
from astroquery.mast import Observations
import glob
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from astropy.table import vstack
from flask import Flask, render_template, request, Response, make_response, send_file, jsonify
import requests
import tensorflow as tf
import base64
import io
from io import BytesIO
import zipfile
import tempfile
import random
import json
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
obslist = []
say = 1

def get_observation_ids(target_id,user_input_id, base_url,obslist):
    # Query MAST for observations
    global lc_products
    print("--->", user_input_id)
    obs_table0 = Observations.query_object(target_id)
    obs_table1 = obs_table0.group_by(['dataproduct_type', 'obs_collection'])
    mask0 = (obs_table1['dataproduct_type'] == 'timeseries') & (obs_table1['obs_collection'] == 'TESS')
    obs_table = obs_table1[mask0]   
    if len(obs_table) > 0:
      obs_ids = obs_table['obsid']
      #print("=======tarid==========", target_id)
      data_products = Observations.get_product_list(obs_table)
      lc_products = data_products[data_products['productSubGroupDescription'] == 'LC']
      #for rw in lc_products['obsID'][:1]: #------> to try first observation when debuging.
      for rw in lc_products['obsID']:
        all_false = np.all(rw == False)
        if not all_false:
          fits_path0 = lc_products['dataURI'][lc_products['obsID'] == rw][0]
          fits_path = base_url + fits_path0
          response = requests.get(fits_path)    
      # Check if the request was successful
          if response.status_code == 200:
            # Open the FITS file from the file-like object
            fits_data = BytesIO(response.content)
            with fits.open(fits_data) as hdul:
                # Access and manipulate the FITS data as needed
                hdr = hdul[0].header
                hedid = hdr['TICID']
                tarid = target_id[3:]
                if hedid == int(tarid):
                  obslist.append(rw)
                  #print(obslist)
      maskob = np.isin(obs_table['obsid'], obslist)
      obs_ids2 = obs_table[maskob]['obsid']
      return obs_ids2
    else:
      obs_ids2 = np.array([])
      return obs_ids2


def load_image_into_numpy_array(fpngname):
  img_data = tf.io.gfile.GFile(fpngname, 'rb').read()
  image = Image.open(BytesIO(img_data))
  #(im_width, im_height) = image.size
  #return np.array(image.getdata()).reshape(
  #    (im_height, im_width, 1)).astype(np.uint8)
  image = np.array(image)
  if len(image.shape) == 2:
        # Expand grayscale image to have 3 channels (RGB equivalent)
        image = np.expand_dims(image, axis=-1)
  elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA images to RGB
        image = image[:, :, :3]
  return image.astype(np.uint8)


def run_inference_for_rcnn(model, fpngname, category_index, conf_th):
    #global ymin, xmin, ymax, xmax, input_width, input_height
    image_np = load_image_into_numpy_array(fpngname)
    input_height, input_width, _ = image_np.shape
    #print("rcnn image shape:",input_height, input_width)
    #image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    #print("---------------------------->", output_dict['detection_classes'])
    # Handle models with masks:
    if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                 image.shape[0], image.shape[1])      
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                         tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    detected_objects = []
    min_conf = float(conf_th)

    for i in range(len(output_dict['detection_scores'])):
      score = output_dict['detection_scores'][i]
      #print("score:", score)
      if score > min_conf:
          
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
          height, width, _ = image_np.shape
          ymin, xmin, ymax, xmax = ymin * height, xmin * width, ymax * height, xmax * width
            #score = score[i]
          box = (xmin, ymin, xmax, ymax)
          if class_name == "pul":
            class_namex = 0
          if class_name == "min":
            class_namex = 1
          detected_object = {
          'bbox': box,  # Convert to list for JSON serialization
          'confidence': score,
          'class_id': class_namex
          }
          #print(detected_object)
          detected_objects.append(detected_object)
    return detected_objects

def detect_objects(image,model_op,conf_th,fpngname):
    global input_width, input_height
    with open("assets/labelmap.pbtxt", 'r') as f:
      labels = [line.strip() for line in f.readlines()]
    if model_op == "SSD" or model_op == "Eff":
      if model_op == "SSD":
        interpreter = tf.lite.Interpreter(model_path="models/ssd/detect.tflite")
      elif model_op == "Eff":
        interpreter = tf.lite.Interpreter(model_path="models/effdet/eff_detect.tflite")
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      input_height = input_details[0]['shape'][1]
      input_width = input_details[0]['shape'][2]
      min_conf = float(conf_th)
      # Preprocess the image
      rgb_image = image.convert("RGB")
      resized_image = rgb_image.resize((input_width, input_height))  # Resize the image to match model input size
      input_data = np.array(resized_image)
      input_data = np.expand_dims(input_data, axis=0)
      input_data = input_data.astype(np.float32) / 255.0  # Normalize pixel values
      interpreter.set_tensor(input_details[0]['index'], input_data)
      interpreter.invoke()
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
      detected_objects = []
        # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
          ymin = int(max(1,(boxes[i][0] * input_height)))
          xmin = int(max(1,(boxes[i][1] * input_width)))
          ymax = int(min(input_height,(boxes[i][2] * input_height)))
          xmax = int(min(input_width,(boxes[i][3] * input_width)))
          object_name = int(classes[i])
          #print("-----------------------------", int(classes[i]))
          score = scores[i]
          box = (xmin, ymin, xmax, ymax)
          #detected_object = object_name, xmin, ymin, xmax, ymax, score
          detected_object = {
              'bbox': box,  # Convert to list for JSON serialization
              'confidence': score,
              'class_id': object_name
          }
          detected_objects.append(detected_object)
      #print(fpngname,"-------- detected_objects---- ->",detected_objects)
      return detected_objects, input_width, input_height

    if model_op=="frcnn":
      #print("------------------------------>",fpngname)
      image_np = load_image_into_numpy_array(fpngname)
      input_height, input_width, _ = image_np.shape
      labelmap_path = os.path.join(os.getcwd(), 'assets/labelmap_frcnn.pbtxt')
      category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
      sfolder_path = os.path.join(os.getcwd(), 'models/frcnn/saved_model')
      modrcnn = tf.saved_model.load(sfolder_path)
      conf_th = conf_th

      detected_objects = run_inference_for_rcnn(modrcnn, fpngname, category_index, conf_th)
      
    return detected_objects, input_width, input_height


def plot_png(tpdata2, pt1, pt2, nb, target_id):
  d_size = 240
  sizep = 5 * 72 / d_size  # pointsize is 5 pxl
  ax = plt.gca()
  ax.scatter(tpdata2['TIME'], tpdata2['SAP_MAG'], marker=",", lw=0, c='Black', s=sizep)
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  tpdata4 = tpdata2[(tpdata2['TIME'] >= pt1) & (tpdata2['TIME'] <= pt2)]
  #print(tpdata4)
  tpdata4 = tpdata4.dropna()  
  if not tpdata4.empty:
    ymin = tpdata4['SAP_MAG'].min()
    ymax = tpdata4['SAP_MAG'].max()
    #print('ymin: ', ymin, 'ymax: ', ymax)
    yaxmin = ymin - (ymax -ymin) / 50 
    yaxmax = ymax + (ymax -ymin) / 50
    ax.set_ylim(yaxmin, yaxmax)
    ax.invert_yaxis()
    ax.set_xlim(pt1, pt2)
    for line in ax.get_lines():
      line.set_color('gray')
    plt.gcf().set_size_inches(d_size / plt.gcf().dpi, d_size / plt.gcf().dpi)
    filepng = '{}_{}_{}.png'.format(target_id,indx,nb)
    plt.savefig(filepng, dpi=plt.gcf().dpi)
    plt.close()
    
    return filepng



def get_fits_path(target_id, selected_obs_id, period_input, base_url, period_factor,model_op,conf_th):
    global nb
    fits_pathf0 = lc_products['dataURI'][lc_products['obsID'] == selected_obs_id][0]
    fits_pathf00 = lc_products['dataURI'][lc_products['obsID'] == selected_obs_id]
    fits_pathf = base_url + fits_pathf0
    data = fits.getdata(fits_pathf, ext=1)
    data2 = pd.DataFrame(np.array(data).byteswap().newbyteorder())
    tpdata = pd.DataFrame(data2, columns=['TIME','SAP_FLUX'])
    tpdata = tpdata.dropna()
    logtemp = np.log10(tpdata.loc[:,'SAP_FLUX']) * -2.5
    tpdata2 = pd.concat([tpdata,logtemp], axis=1)
    tpdata2.columns = ['TIME','SAP_FLUX', 'SAP_MAG']
    conf_calc = pd.DataFrame()
    s_interval = float(period_factor) * float(period_input)
    if ((tpdata2['TIME'].max() - tpdata2['TIME'].min()) / s_interval) > 1:
      for nb in range(int((tpdata2['TIME'].max() - tpdata2['TIME'].min()) / s_interval)):
        pt1 = tpdata2['TIME'].min() + (nb * s_interval)
        pt2 = tpdata2['TIME'].min() + ((nb + 1) * s_interval)
        if pt2 < tpdata2['TIME'].max():
            fpngname = plot_png(tpdata2, pt1, pt2, nb, target_id)
            if fpngname is not None:
              with open(fpngname, 'rb') as file:
                image_bytes = file.read()
              image = Image.open(BytesIO(image_bytes))
              detection, input_width, input_height = detect_objects(image,model_op,conf_th,fpngname)
              class_ids = {item['class_id'] for item in detection}
              if {0, 1}.issubset(class_ids):
                confidences = [entry['confidence'] for entry in detection]
                average_confidence = sum(confidences) / len(confidences)
                confcalc0 = pd.DataFrame({'filename': [fpngname], 'conf_avg': [average_confidence]})
                conf_calc = pd.concat([conf_calc, confcalc0], ignore_index=True)
      return detection, conf_calc

    else:
      detection = 505
      conf_calc = 505
      return detection, conf_calc



def remove_tuples_with_common_last_elements(input_list):
    seen_last_elements = set()
    unique_tuples = []
    for tup in input_list:
        last_element = tup[3]
        if last_element not in seen_last_elements:
            unique_tuples.append(tup)
            seen_last_elements.add(last_element)
    return unique_tuples



def print_png_elements(tuples_list):
    for tup in tuples_list:
        for element in tup:
            if isinstance(element, str) and ".png" in element:
                print(element)



def above_conf_detection(target_id, kis, period_input,base_url,period_factor,model_op,conf_th):
  global fpname, encoded_image, indx
  result_above0 = []
  to_best = pd.DataFrame()
  for indx in kis:
    #print("-------------")
    #print("indx:",indx)
    sy = 1
    detection2, conf_to_best = get_fits_path(target_id, indx, period_input,base_url, period_factor, model_op, conf_th)
    if detection2 != 505:
      to_best = pd.concat([to_best,conf_to_best], ignore_index=True)     
      if not to_best.empty:
        best_df = to_best[to_best['conf_avg'] >= float(conf_th)]
        for fpname in best_df['filename']:
          with open(fpname, 'rb') as file2:
               image_bytes = file2.read()
          image2 = Image.open(BytesIO(image_bytes))
          detection2, input_width, input_height = detect_objects(image2,model_op,conf_th,fpname)
          #print(fpname,"------------- detected_objects --------->", detection2)
          #print("detection2:", detection2)
          dfdet2 = pd.DataFrame(detection2)
          if not dfdet2.empty: 
            sy +=1
            dfdet2['Class'] = dfdet2['class_id'].apply(lambda x: 'M' if x == 1 else 'P')
            dfdet3 = dfdet2[['Class', 'confidence']]
            dfdet3 = dfdet3.copy()
            dfdet3['confidence'] = dfdet3['confidence'].round(3).astype(str)         
            for obj in detection2:
              xmin, ymin, xmax, ymax = obj['bbox']
              ##### calculate average confidences and plot image with highest ####### 
              label = f"{obj['class_id']} ({obj['confidence']:.2f})"
              draw = ImageDraw.Draw(image2)              
              xmin, ymin, xmax, ymax = xmin * (240/input_width), ymin * (240/input_height), xmax * (240/input_width), ymax * (240/input_height)
              if obj['class_id'] == 0:
                draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
                idp, x1p, y1p, x2p, y2p = obj['class_id'], xmin, ymin, xmax, ymax
              elif obj['class_id'] == 1:
                draw.rectangle([xmin, ymin, xmax, ymax], outline='blue', width=2)
                idm, x1m, y1m, x2m, y2m = obj['class_id'], xmin, ymin, xmax, ymax
            # Convert annotated image to base64 encoded string
            buffered = BytesIO()
            image2.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result0 = detection2, dfdet3, encoded_image, fpname, indx, input_width, input_height
            result_above0.append(result0)
            #print("result_above0:", result_above0)
  #print("result_above0:", result_above0)
  result_above = remove_tuples_with_common_last_elements(result_above0)
  #print("result_above:", result_above)
  return result_above



def period_from_tebs(target_id):
  dtebs0 = pd.read_csv("assets/tess_ebs_data.csv")
  dtebs = dtebs0[~dtebs0['target'].str.contains('_')]
  if not dtebs.loc[dtebs['target'] == target_id].empty:
    period_input = dtebs.loc[dtebs['target'] == target_id , 'period'].values[0]
  else:
    period_input = 404
  return period_input



def input_extention(user_input_id,period_input,period_factor,conf_th,model_op,say,result_data):
      #print(user_input_id,period_input,period_factor,conf_th,model_op,say)
      #result_data = []
      kis0 = get_observation_ids(user_input_id, user_input_id, base_url, obslist)
      if kis0.size > 0:  
        kis = sorted(kis0)
        #print(kis)
        #result_data = []
        detco = []
        filnm = []
        show_pd = 2    
        result11 = above_conf_detection(user_input_id, kis, period_input,base_url,period_factor,model_op,conf_th)
        if result11 != None:
        #print('=== result11 ===')
        #print(result11)
          for litem in result11:
            #print("=== litem ===")
            #print(litem)
            detec, dfdet3, encoded_image, fpname, indx, input_width, input_height = litem
            for itek in detec:
              itek['bbox'] = list(itek['bbox'])
            for itex in detec:
              itex['file'] = fpname
            detecj0 = [{k: v for k, v in detec.items() if k != 'confidence'} for detec in detec]
            detecj2 = {"res": detecj0}
            filnm.append(fpname)
            filnm = [string.replace("'", "") for string in filnm]
            if dfdet3.empty:
              dfdet3_em = True
            else:
              dfdet3_em = False
            detecj = json.dumps(detecj2)
            detecString_json = detecj.replace("'", '"')
            nfil = "ext_" + fpname 
            shutil.copy(fpname, nfil)
            result_data.append({
              'user_input': user_input_id,
              'detec': detec,
              'show_pd': show_pd,
              'period_input': period_input,
              'period_factor': period_factor,
              'conf_th': conf_th,
              'model_op': model_op,
              'image_data': encoded_image,
              'dfdet3': dfdet3,
              'dfdet3_em': dfdet3_em,
              'filnm': filnm,
              'fpname': fpname,
              'detco' : detco,
              'detecString_json': detecString_json,
              'indx': indx,
              'say': say,
              'wid': input_width,
              'hei': input_height
            })
            say += 1
      #print(result_data)  
      files_to_remove = glob.glob('TIC*.png')
      for file_rm in files_to_remove:
        os.remove(file_rm)
      
      return result_data



def input_f():
    if 'user_input' in request.form:
      #say = 1
      print("========================SINGLE INPUT=================")
      user_input_id = request.form['user_input']
      file_up = None  # No file upload
      if 'disable_radio' in request.form:
        period_input = period_from_tebs(user_input_id)  
        if period_input == 404:
          result_data = []
          result_data.append({'period_input': period_input})
          return render_template('index.html', result_data=result_data)
      else:
        period_input = float(request.form.get('period_input', ''))
      period_factor = request.form['period_factor']
      conf_th = request.form['conf_th']
      model_op = request.form['radio_option']
      result_data = []
      result_data = input_extention(user_input_id,period_input,period_factor,conf_th,model_op,say,result_data)
      #return render_template('index.html', result_data=result_data)
      return result_data

    elif 'fileUpload' in request.files:
        #say = 1
        print("========================FILE UPLOAD=================")
        #result_data1 = []
        #show_pd = 2
        file_up = request.files['fileUpload']
        user_input_id = None  # No form input
        filedf = pd.read_csv(file_up)
        filedf.columns = ['target', 'per', 'per_fac']
        #print(filedf)
        result_data = []
        for index, row in filedf.iterrows():
          user_input_id = row['target']
          period_input = row['per']
          period_factor = row['per_fac']
          conf_th = request.form['conf_th']
          model_op = request.form['radio_option']
          result_data = input_extention(user_input_id,period_input,period_factor,conf_th,model_op,say,result_data)
          #return render_template('index.html', result_data=result_data)
          #result_data1.append(result_data)
        #print(result_data1)
        return result_data



def response_f():
    response = send_file('index.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response



def downl_f():
      try:
        global rnd
        rnd = random.randint(100, 999)
        data = request.get_json()
        fnm = data.get('fnm')  # Get the filename
        base_name, extension = os.path.splitext(fnm)
        fnm2 = f"{base_name}_{rnd}{extension}"
        #print("=============", rnd, fnm2)
        lines = data.get('lines')  # Get the file content
        # Write the lines to a text file
        with open(fnm, 'w') as file:
            file.write(lines)
        text_response = send_file(fnm, as_attachment=True)#, download_name=imnm2)
        # Return the text file response
        return text_response
      except Exception as e:
        # Log the error message
        print(e)
        # Return a 500 Internal Server Error response with the error message
        return jsonify({'error': str(e)}), 500



def downim_f():
    try:
        imnm = request.args.get('imnm')
        base_name, extension = os.path.splitext(imnm)
        imnm2 = f"{base_name}_{rnd}{extension}"
        #print("=============",rnd,imnm2)
        # Assuming fpname is the path to the image file
        return send_file(imnm, as_attachment=True)#, download_name=imnm2)
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to download image'}), 500