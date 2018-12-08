
# coding: utf-8

import pandas as pd
import numpy as np
import datetime 
import subprocess
import sys
import os
from dateutil.relativedelta import relativedelta
import getopt

def execute_time_window(date_end_str, plidata, fire_pre14, fire_new):

    year, month, day = [int(i) for i in date_end_str.split('-')]
    date_end = datetime.datetime(year = year, month = month, day = day)

    # Start date of time window 
    date_start = date_end - relativedelta(months = 8 * 12) 
    date_start_str = date_start.strftime("%Y-%m-%d")
    
    # Obatain timewindow
    # For pli dataset
    plidata_w = plidata[(pd.to_datetime(plidata['INSPECTION_DATE']) < date_end) &
                        (pd.to_datetime(plidata['INSPECTION_DATE']) > date_start)]

    # For fire_pre14 dataset
    fire_pre14_w = fire_pre14

    # For fire_new dataset
    fire_new_w = fire_new[(pd.to_datetime(fire_new['CALL_CREATED_DATE']) < date_end)]

    # Save dataset
    # NOTE: Create the following folders if not exist
    plidata_w.to_csv("datasets_assess/pli_end_{0}.csv".format(date_end_str), index=False)
    fire_pre14_w.to_csv("datasets_assess/Fire_Incidents_Pre14_end_{0}.csv".format(date_end_str), index=False)
    fire_new_w.to_csv("datasets_assess/Fire_Incidents_New_end_{0}.csv".format(date_end_str), index=False)

    # Execute riskmodel.py
    if Verbose:
        # print("Processing time window from {} to {}".format(date_start_str, date_end_str))
        print("Processing time window end at {}".format(date_end_str))

    data_processor(date_end_str)

    if Verbose:
        # print("Finished time window from {} to {}".format(date_start_str, date_end_str))
        print("Finished time window end at {}".format(date_end_str))

    # Delete created sub dataset to release disk
    os.remove("datasets_assess/pli_end_{0}.csv".format(date_end_str))
    os.remove("datasets_assess/Fire_Incidents_Pre14_end_{0}.csv".format(date_end_str))
    os.remove("datasets_assess/Fire_Incidents_New_end_{0}.csv".format(date_end_str))
    
# Execute code
def data_processor(date_end_str):

    try:
        args = ["python -W ignore risk_model_residential_kdd.py {0}".format(date_end_str)]
        
        p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        if(stderr and p.returncode):
            sys.exit(stderr.decode("utf-8"))            

    except Exception as e:
        sys.exit(e)


if __name__ == "__main__":
    # create directory paths for opening files
    # NOTE: Create the following folders if not exist
    curr_path = os.getcwd()
    dataset_path = os.path.join(curr_path, "datasets/")

    # Reading pli dataset
    plidata = pd.read_csv(os.path.join(dataset_path, "pli.csv"),encoding = 'utf-8',dtype={'STREET_NUM':'str','STREET_NAME':'str'}, low_memory=False)
    # Reading city of Pittsburgh dataset
    # pittdata = pd.read_csv(os.path.join(dataset_path, "pittdata.csv"),encoding = "ISO-8859-1", dtype={'PROPERTYADDRESS':'str','PROPERTYHOUSENUM':'str','CLASSDESC':'str'}, low_memory=False)

    # Reading fire incident dataset
    fire_pre14 = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_Pre14.csv"),encoding = 'latin-1',dtype={'street':'str','number':'str'}, low_memory=False)
    fire_new = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_New.csv"),encoding = 'utf-8',dtype={'street':'str','number':'str'}, low_memory=False)

    # Get initial time window's end date, number of windows, verbose
    try:     

        options, _ = getopt.getopt(sys.argv[1:], "vd:n:")

        init_date_end_str = ""
        num = 0

        global Verbose
        Verbose = False

        for opt, arg in options:
            if opt == "-v":
                Verbose = True

            elif opt == "-d":
                init_date_end_str = arg
                year, month, day = [int(i) for i in init_date_end_str.split('-')]
                init_date_end = datetime.datetime(year = year, # Convert to datetime obj
                                                  month = month, 
                                                  day = day) 
                if (init_date_end > datetime.datetime.now()):
                    exit("Input future date.")

            elif opt == "-n":
                num = int(arg)
                if (num <= 0):
                    exit("wrong input number")

    except Exception as e: 
        sys.exit(e)
        
    
    # Execute each time window
    for i in range(num):
        date_end = init_date_end - relativedelta(weeks = 1 * i)
        date_end_str = date_end.strftime("%Y-%m-%d")
        
        if(Verbose):
            print("=============================================================")

        execute_time_window(date_end_str, plidata, fire_pre14, fire_new)


