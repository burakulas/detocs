"""
Author: Burak Ulas -  github.com/burakulas
2024, Konkoly Observatory and COMU
"""

import pandas as pd
import numpy as np
import os, sys
import shutil
from astroquery.mast import Observations
import glob
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread will likely fail.")
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from astropy.table import vstack
from flask import Flask, render_template, request, Response, make_response
import requests
from io import BytesIO
import tensorflow as tf
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from flask import Flask, send_file, jsonify
import matplotlib.patches as patches
from datetime import datetime
from astroquery.simbad import Simbad


base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
dir_name = "detected" + "_" + datetime.today().strftime('%Y-%m-%d_%H-%M')
os.mkdir(dir_name)

obslist = []


def get_observation_ids(target_id):
    # Query MAST for observations
    global lc_products
    print("--->", user_input_id)
    obs_table0 = Observations.query_object(target_id)
    obs_table1 = obs_table0.group_by(['dataproduct_type', 'obs_collection'])
    mask0 = (obs_table1['dataproduct_type'] == 'timeseries') & ((obs_table1['obs_collection'] == 'Kepler') | (obs_table1['obs_collection'] == 'K2'))
    obs_table = obs_table1[mask0]
    if len(obs_table) > 0:
      obs_ids = obs_table['obsid']
      data_products = Observations.get_product_list(obs_table)
      lc_products = data_products[(data_products['productSubGroupDescription'] == 'SLC')]
      for rw in np.unique(lc_products['obsID']):
        all_false = np.all(rw == False)
        if not all_false:
          fits_path0 = lc_products['dataURI'][lc_products['obsID'] == rw][0]
          fits_path = base_url + fits_path0
          response = requests.get(fits_path)
          if response.status_code == 200:
            fits_data = BytesIO(response.content)
            with fits.open(fits_data) as hdul:
                hdr = hdul[0].header
                hedid = hdr['OBJECT']
                if target_id[3:].startswith('0'):
                  zerost0 = target_id[3:]
                  zerost = zerost0[1:]
                  tarid = target_id[:3] + " " +zerost
                else:
                  tarid = target_id[:3] + " " +target_id[3:]
                if hedid == str(tarid):

                  print("-> Object in .fits header and target match in obsID:", rw)
                  obslist.append(rw)

      
      maskob = np.isin(obs_table['obsid'], obslist)
      obs_ids2 = obs_table[maskob]['obsid']

      return obs_ids2
    else:
      obs_ids2 = np.array([])

      return obs_ids2

def get_simbad(target_id):
  custom_simbad = Simbad()
  custom_simbad.add_votable_fields('otypes')
  result_table = custom_simbad.query_object(target_id)
  if result_table:
    otyp = result_table['OTYPES'][0]
  return otyp


def detect_objects(image):
    global input_width, input_height
    with open("labelmap.pbtxt", 'r') as f:
      labels = [line.strip() for line in f.readlines()]

    interpreter = tf.lite.Interpreter(model_path="SSD_detect.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    min_conf = float(conf_th)
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

    for i in range(len(scores)):
      if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
        ymin = int(max(1,(boxes[i][0] * input_height)))
        xmin = int(max(1,(boxes[i][1] * input_width)))
        ymax = int(min(input_height,(boxes[i][2] * input_height)))
        xmax = int(min(input_width,(boxes[i][3] * input_width)))
        object_name = int(classes[i])
        score = scores[i]
        box = (xmin, ymin, xmax, ymax)
        detected_object = {
            'bbox': box,  # Convert to list for JSON serialization
            'confidence': score,
            'class_id': object_name
        }
        detected_objects.append(detected_object)
    return detected_objects

def plot_png(tpdata2, pt1, pt2, nb, target_id):
  global yaxmin, yaxmax
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
    filepng = '{}_{}_{}_{}_{}.png'.format(target_id,indx,nb,sy,nff)
    #print(filepng, pt1, pt2, yaxmin, yaxmax)
    plt.savefig(filepng, dpi=plt.gcf().dpi)
    plt.close()
    detect_png(filepng)


def detect_png(filepng):
    global sumfile , new_dir, smf
    with open(filepng, 'rb') as file:
      image_bytes = file.read()

    image = Image.open(BytesIO(image_bytes))
    detection = detect_objects(image)

    class_ids = {item['class_id'] for item in detection}
    
    if {0, 1}.issubset(class_ids):
      new_dir = os.path.join(dir_name, str(user_input_id))
      if not os.path.exists(new_dir):
          os.makedirs(new_dir)
      confidences = [entry['confidence'] for entry in detection]
      for obj in detection:
        xcls = obj['class_id']
        label = f"{obj['class_id']} ({obj['confidence']:.2f})"
        label0 = f"{obj['class_id']} {obj['bbox']} ({obj['confidence']:.2f})"
        sumfile = dir_name +"/summary_" + str(user_input_id) + ".txt"
        smf = 202
        wrtn = str(filepng) + ":" + str(label0)
        with open(sumfile, 'a') as sfile:
          sfile.write(wrtn + '\n')
        #print("---------->", filepng, obj)
        xmin, ymin, xmax, ymax = obj['bbox']
        xmin, ymin, xmax, ymax = xmin * (240/input_width), ymin * (240/input_height), xmax * (240/input_width), ymax * (240/input_height)
        plot_anno(filepng, label, xmin, ymin, xmax, ymax, xcls)
      otp = get_simbad(user_input_id)
      otp_el = otp.split('|')
      unique_otp = list(set(otp_el))
      unique_otp2 = ', '.join(unique_otp)
      titler = str(user_input_id)
      plt.gca().axes.xaxis.set_visible(False)
      plt.gca().axes.yaxis.set_visible(False)
      plt.gca().spines['top'].set_visible(False)
      plt.gca().spines['right'].set_visible(False)
      plt.gca().spines['left'].set_visible(False)
      plt.gca().spines['bottom'].set_visible(False)
      plt.gca().set_title(titler, fontsize=15) #  <----------------make it titl (line209) for simbad otype as title
      plt.savefig("{}/det_{}".format(new_dir,filepng),dpi=300)
      plt.close()
  
    else:
      smf = 404


def plot_anno(filepng, label, xmin, ymin, xmax, ymax, xcls):

    image = np.array(Image.open(filepng)) 
    plt.imshow(image)

    if xcls == 0:
         rectangle0 = [(xmin, ymin), (xmax, ymax)]
         rect0 = patches.Rectangle(rectangle0[0], rectangle0[1][0] - rectangle0[0][0], rectangle0[1][1] - rectangle0[0][1], linewidth=2, edgecolor='r', facecolor='none')
         plt.gca().add_patch(rect0)
         y_offset = 3  # Adjust as needed
         formatted_valuep = "P: {}".format(label[3:7])
         value_positionp = (xmin, ymin - y_offset)
         plt.text(value_positionp[0], value_positionp[1], formatted_valuep, color='red', fontsize=15)

    elif xcls == 1:
         rectangle1 = [(xmin, ymin), (xmax, ymax)]
         rect1 = patches.Rectangle(rectangle1[0], rectangle1[1][0] - rectangle1[0][0], rectangle1[1][1] - rectangle1[0][1], linewidth=2, edgecolor='b', facecolor='none')
         plt.gca().add_patch(rect1)
         y_offset = 3  # Adjust as needed
         formatted_valuem = "M: {}".format(label[3:7])
         value_positionm = (xmin, ymin - y_offset)
         plt.text(value_positionm[0], value_positionm[1], formatted_valuem, color='blue', fontsize=15)
    

def get_fits_path(target_id, selected_obs_id, period_input,fits_pathf):
    global nb,nff

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
      nff = 1
      for nb in range(int((tpdata2['TIME'].max() - tpdata2['TIME'].min()) / s_interval)):
        
        pt1 = tpdata2['TIME'].min() + (nb * s_interval)
        pt2 = tpdata2['TIME'].min() + ((nb + 1) * s_interval)
       
        if pt2 < tpdata2['TIME'].max():
            fpngname = plot_png(tpdata2, pt1, pt2, nb, target_id)
            nff += 1      


    else:
      detection = 505
      conf_calc = 505


def above_conf_detection(target_id, kis, period_input):
  global fpname, encoded_image, indx, sy
  result_above0 = []
  to_best = pd.DataFrame()
  for indx in kis:
    fits_pathf0 = lc_products['dataURI'][lc_products['obsID'] == indx]
    sy = 1
    for flong in range(len(fits_pathf0)):
      fits_pathf = base_url + fits_pathf0[flong]
      get_fits_path(target_id, indx, period_input, fits_pathf)
      sy += 1
  print("->",user_input_id, "finished")
  print(" ")


file_up = "alexi_kic_try.csv"
filedf = pd.read_csv(file_up)
filedf.columns = ['target', 'per', 'per_fac', 'conf_th', 'model']
print(filedf)
print(" ")
for index, row in filedf.iterrows():
  user_input_id = row['target']
  period_input = row['per']
  period_factor = row['per_fac']
  conf_th = row['conf_th']
  model_op = row['model']
  kis0 = get_observation_ids(user_input_id)
  if kis0.size > 0:
    kis = sorted(kis0)
    detco = []
    filnm = []
    show_pd = 2
    result11 = above_conf_detection(user_input_id, kis, period_input)
    #print("SMF:", smf)
    #if smf == 202:
    #  shutil.move(sumfile, dir_name)


    for filename in os.listdir(os.getcwd()):
      if filename.startswith("KIC") and filename.endswith(".png"):
        os.remove(filename)





