import configparser
import json
import os
import sys
import time
import logging
import biomarker_discovery_script_selection_backend as model_training
import psycopg2
import datetime
import shutil
import warnings

from multiprocessing import set_start_method


def start_next_job():
    """
    Searched db for the next available job in the queue and perform Biomarkers Training Multibiomarker Predictive
    Analytics Model
    :return: nothing just end the script
    """
    conn, config, thisProcessID = config_connect_db()
    # Find next job to calculate
    newjob = find_next_job(conn, config, thisProcessID)
    # Set job running
    set_job_running(newjob[0], conn, newjob[1], thisProcessID)
    # Process current job and end accordingly
    result = process_job(conn, config, newjob, thisProcessID)
    if result[0]:
        # Everything successful
        set_job_completed(newjob[0], conn, result[1], newjob[1], thisProcessID)
        logging.info(
            "PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Training Multibiomarker Predictive Analytics Model process has "
            "finished successfully".format(thisProcessID, newjob[0], newjob[1]))
        sys.exit("Training Multibiomarker Predictive Analytics Model process has finished successfully")
    else:
        # Encountered an error
        set_job_error(newjob[0], conn, newjob[1], result=result[1], pid=thisProcessID)
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Identification Pipeline process has finished "
                     "unsuccessfully".format(thisProcessID, newjob[0], newjob[1]))
        sys.exit("Training Multibiomarker Predictive Analytics Model process has finished unsuccessfully")


# Find insybio.ini file to config the program and connect to Database
def config_connect_db():
    """
    Get configurations from ini file, and connect to the data base.
    :return: conn: connection object of db, thisProcessID: current pid, config: configuration dictionary
    """
    config = configparser.ConfigParser()
    scriptPath = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
    scriptParentPath = os.path.abspath(os.path.join(scriptPath, os.pardir))
    configParentPath = os.path.abspath(os.path.join(scriptParentPath, os.pardir))
    config.read(configParentPath + '/insybio.ini')

    # for regular use, use logs directory from ini
    logs_path = config["logs"]["logpath"]

    # Init logging
    todaystr = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(filename="{}biomarkers_reports_{}.log".format(logs_path, todaystr),
                        level=logging.INFO, format='%(asctime)s\t %(levelname)s\t%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # read process ID
    thisProcessID = os.getpid()

    # connect to database
    try:
        conn = psycopg2.connect("dbname='{}' user='{}' host='{}' password='{}'".format(config['insybio.db']['dbname'],
                                                                                       config['insybio.db'][
                                                                                           'dbusername'],
                                                                                       config['insybio.db']['dbhost'],
                                                                                       config['insybio.db'][
                                                                                           'dbpassword']))
        conn.autocommit = True
    except psycopg2.Error:
        logging.exception("PID:{}\tJOB:\tUSER:\tNo connection to db".format(thisProcessID))
        sys.exit("No connection to db")
    return conn, config, thisProcessID


def find_next_job(conn, config, pid=0):
    """
    Find next job with status 1 and type 4 = model Trainig
    :param conn: db connection object
    :param pid: job's pid
    :param config: configuration dictionary
    :return: newjob: list with job_id, job_user and job's input file path
    """
    # define cursor
    cur = conn.cursor()

    # check if there are running jobs
    query = "SELECT count(id) as model_training from biomarkers_jobs WHERE status=2 AND type=4"
    cur.execute(query)
    numRunningJobs = cur.fetchone()
    # Check if number of running jobs is grater that what the ini specifies uncomment in production
    if numRunningJobs[0] >= int(config['biomarkers.modeltraining']['paralleljobs']):
        # logging.info("There are still running jobs")
        sys.exit()

    # get next job to run
    try:
        cur.execute("""SELECT id,\"user\",input from biomarkers_jobs WHERE status=1 AND type=4 ORDER BY
         starttimestamp ASC LIMIT 1""")   # selecting columns id, user, input
        newjob = cur.fetchone()
    except Exception:
        # logging.info("No jobs found")
        sys.exit()

    # if no jobs found end else return newjob
    if newjob is None:
        sys.exit()
    return newjob


# Set current job to running
def set_job_running(job_id, conn, job_user='unknown', pid=0):
    """
    updates db that the job processing started, set status = 2,
    and startruntimestamp to current timestamp
    :param job_id: job's id
    :param conn: db connection object
    :param job_user: job's user
    :param pid: job's pid
    :return: True
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update that the job processing started".format(pid, job_id, job_user))
    timestamp = int(time.time())
    cur = conn.cursor()
    query = "UPDATE biomarkers_jobs SET status = 2, startruntimestamp = %s WHERE id = %s"
    cur.execute(query, (timestamp, job_id))
    set_jobs_pid(job_id, conn, job_user, pid)
    return True


def set_jobs_pid(jobid, conn, user, pid):
    """
    Add job's PID to pipeline_job_processes table
    :param jobid: this job's id
    :param conn: Database connection object
    :param user: job's user
    :param pid: job's PID
    :return: True if insert was successful
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update current job's PID".format(pid, jobid, user))
    cur = conn.cursor()
    query = "INSERT INTO biomarkers_jobs_processes (pid, ip, job) VALUES (%s, %s, %s);"
    cur.execute(query, (pid, 'localhost', jobid))
    return True


# Set current job to completed successfully

def set_job_completed(job_id, conn, result_json, job_user='unknown', pid=0):
    """
    updates db that the job processing completed succesfully, set status = 3,
    and endruntimestamp to current timestamp
    :param job_id: job's id
    :param conn: db connection object
    :param result_json: dictionary of output file and messages
    :param job_user: job's user
    :param pid: job's pid
    :return: True
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update that the job processing completed".format(pid, job_id, job_user))
    timestamp = int(time.time())
    result = json.dumps(result_json)
    cur = conn.cursor()
    query = "UPDATE biomarkers_jobs SET status = 3, endtimestamp = %s, result = %s WHERE id = %s"
    cur.execute(query, (timestamp, result, job_id))
    return True


# Set current job to completed with error

def set_job_error(job_id, conn, job_user='unknown', pid=0, result={"error": "Unknown error."}):
    """
    updates db that the job processing completed unsuccessfully, set status = 4,
    and startruntimestamp and endtimestamp to current timestamp
    :param job_id: job's id
    :param conn: db connection object
    :param job_user: job's user
    :param pid: job's pid
    :param result: the error message in json that should be updated
    :return: True
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update that the job encountered an error".format(pid, job_id, job_user))
    timestamp = int(time.time())
    # result = "{\"error\": \"Either miRNA or Targets are not existing.\"}"
    cur = conn.cursor()
    query = "UPDATE biomarkers_jobs SET status = 4, endtimestamp = %s, result = %s WHERE id = %s"
    cur.execute(query, (timestamp, json.dumps(result), job_id))
    return True


def update_ds(filepath, conn, description, filetype, fileformat, job_user='unknown', jobid=0, pid=0):
    """
    Inserts the new biological file in the datastore
    :param filepath: the absolute path to the new file
    :param conn: db connection object
    :param description: new file's description
    :param filetype: file's type according to ds_filetypes table
    :param fileformat: file's format according to ds_fileformats table
    :param job_user: job's user id
    :param jobid: job's id
    :param pid: this job's Pid
    :return: datastores new entry id
    """

    timestamp = int(time.time())
    # filepath = Path(file).resolve()
    filesize = os.path.getsize(filepath)
    filename = os.path.basename(filepath)
    targetfile = "/".join((os.path.basename(os.path.dirname(filepath)), filename))
    cur = conn.cursor()
    query = "INSERT INTO ds_files (description, targetfile, create_timestamp, modify_timestamp, filetype, user_id, " \
            "original_name, fileformat, file_size) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;"
    try:
        cur.execute(query, (description, targetfile, timestamp, timestamp, filetype, job_user, filename, fileformat,
                            filesize))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in DB insertion of the new created biological file.".format(
            pid, jobid, job_user))
        return False
    insertid = cur.fetchone()[0]

    insert_query = "INSERT INTO biomarkers_jobs_files (job,file,type) VALUES (%s,%s,%s)"
    try:
        cur.execute(insert_query, (jobid, insertid, 2))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in DB insertion of the new created biological file.".format(
            pid, jobid, job_user))
        return False

    conn.commit()
    return insertid


def process_job(conn, config, newjob, pid):
    """
    Perform Training Model and Store produced file to datastore
    :param conn: Database connection object
    :param config: insybio.ini configuration dictionary
    :param newjob: new job's list of inputs from biomarkers jobs table, with info about job's id user and inputs
    :param pid: this job's PID
    :return: False if encountered error, else True and a output_dictionary with the error message or the
     produced files.
    """

    # get jobs details
    jobid = newjob[0]
    user = newjob[1]
    # parse job parameters
    jobparams = json.loads(newjob[2])
    output_dictionary = {}

    # run process job
    try:
        result = model_training.run_model_and_splitter_selectors(
            jobparams['dataset_filename'], jobparams['labels_filename'], float(jobparams['filtering_percentage']),
            int(jobparams['selection_flag']), int(jobparams['split_dataset_flag']),
            jobparams['goal_significances_string'], jobparams['selected_comorbidities_string'],
            int(jobparams['filetype']), int(jobparams['has_features_header']), int(jobparams['has_samples_header']),
            jobparams['output_folder'], int(jobparams['logged_flag']), int(jobparams['population']),
            int(jobparams['generations']), float(jobparams['mutation_probability']),
            float(jobparams['arithmetic_crossover_probability']), float(jobparams['two_points_crossover_probability']),
            int(jobparams['num_of_folds']), user, jobid, pid, config['insybio.runtime']['thread_num'])
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has occurred during Training Multibiomarker Predictive"
                          " Analytics Model".format(pid, jobid, user))
        output_dictionary['error'] = "An exception has occurred in Train Multibiomarker Predictive Analytics Model: " \
                                     "{}".format(e)
        return [False, output_dictionary]

    # # check if everything went OK
    if int(result[0]) == 0:
        # process went wrong
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been occurred during step 5 of Biomarkers "
                          "Identification Pipeline: {}".format(pid, jobid, user, str(result[1])))
        output_dictionary['error'] = "An exception has been occurred during step 5 of Biomarkers Identification " \
                                     "Pipeline: {}".format(str(result[1]))
        return [False, output_dictionary]

    else:
        # move results to user's DS
        if int(jobparams['selection_flag']) == 2 or int(jobparams['selection_flag']) == 0:
            selected_mq_file_IDs = list()
            selected_biom_file_IDs = list()
            # Storing files in datastore
            selected_mq_files = os.listdir(jobparams['output_folder'] + 'Selected_MQ_files/')
            for file in selected_mq_files:
                file_path = '{}Selected_MQ_files/{}'.format(jobparams['output_folder'], file)
                description = "Selected MQ file created from ({0}) and ({1}), {2}".format(jobparams['dataset_title'],
                                                                                          jobparams['labels_title'],
                                                                                          file[:-4])
                # filetype 7, MQ file, fileformat 2 tsv
                mq_sign_dsid = update_ds(file_path, conn, description, 7, 2, user, jobid, pid)
                selected_mq_file_IDs.append(mq_sign_dsid)

            selected_biom_files = os.listdir(jobparams['output_folder'] + "selected_biomarkers_files/")
            for file in selected_biom_files:
                file_path = '{}selected_biomarkers_files/{}'.format(jobparams['output_folder'], file)
                description = "Selected biomarkers file created from {}".format(file[:-4])
                # filetype 22, biomarkers file, fileformat 2 tsv
                selected_biom_dsid = update_ds(file_path, conn, description, 22, 2, user, jobid, pid)
                selected_biom_file_IDs.append(selected_biom_dsid)
            output_dictionary["Selected_MQ_file_IDs"] = selected_mq_file_IDs
            output_dictionary["Selected_MQ_filenames"] = selected_mq_files

        model_folder = "{}classification_models/".format(jobparams['output_folder'])
        models_archive = "{}models_{}".format(jobparams['output_folder'], jobid)
        shutil.make_archive(models_archive, 'zip', model_folder)

        # model_files = os.listdir(model_folder)
        # with zipfile.ZipFile(models_archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #     for m_file in model_files:
        #         if "pkl.z" in m_file:
        #             zipf.write(os.path.join(model_folder, m_file))

        description = "Models created from ({}) and ({})".format(jobparams['dataset_title'], jobparams['labels_title'])
        # filetype 21, Model file, fileformat 9 ZIP (models are in a zipped folder)
        dsfileID_model = update_ds(models_archive + '.zip', conn, description, 21, 9, user, jobid, pid)
        output_dictionary["model"] = models_archive + '.zip'
        output_dictionary["dsfileID_model"] = dsfileID_model

        if int(jobparams['split_dataset_flag']) == 1:
            testset_file = "{}test_dataset_with_headers.txt".format(jobparams['output_folder'])
            description = "Test dataset with Headers created from ({}) and ({})".format(jobparams['dataset_title'],
                                                                                        jobparams['labels_title'])
            # filetype 19, Dataset file, fileformat 2 tsv
            dsfileID_testset = update_ds(testset_file, conn, description, 19, 2, user, jobid, pid)

            output_dictionary["dataset"] = testset_file
            output_dictionary["dsfileID_testset"] = dsfileID_testset

            testlabel_file = "{}test_labels.txt".format(jobparams['output_folder'])
            description = "Test labels created from ({}) and ({})".format(jobparams['dataset_title'],
                                                                          jobparams['labels_title'])
            # filetype 20, Label file, fileformat 2 tsv
            dsfileID_testlabel = update_ds(testlabel_file, conn, description, 20, 2, user, jobid, pid)

            output_dictionary["test_labels"] = testlabel_file
            output_dictionary["dsfileID_testlabel"] = dsfileID_testlabel

        return [True, output_dictionary]


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    warnings.filterwarnings("ignore")
    start_next_job()

"""

For testing replace the paths of the dataset_filename, labels_filename, and output folder in the next INSERT

INSERT INTO public.biomarkers_jobs VALUES (7, 'info@insybio.com', 15410000, NULL, NULL, 1, 
'{"dataset_filename":"\/home\/milia\/Documents\/Projects\/features-v3.2\/backend-application\/insybio-biomarkers\/
New_Tools\/04.Training_Multibiomarker_Predictive_Analytics_Model\/Input\/twoclass\/output_full_lung_data_0.1_2_1.txt",
"labels_filename":"\/home\/milia\/Documents\/Projects\/features-v3.2\/backend-application\/insybio-biomarkers\/
New_Tools\/04.Training_Multibiomarker_Predictive_Analytics_Model\/Input\/twoclass\/lung_labels.txt",
"filtering_percentage":"0.3","selection_flag":"2","split_dataset_flag":"0","goal_significances_string":"1,10,10,1",
"selected_features_string":"","filetype":"11","has_features_header":"1","has_samples_header":"1",
"output_folder":"\/home\/milia\/Documents\/Projects\/features-v3.2\/backend-application\/insybio-biomarkers\/New_Tools
\/04.Training_Multibiomarker_Predictive_Analytics_Model\/Output\/lung_scheduler\/","population":"50",
"generations":"100","mutation_probability":"0.01","arithmetic_crossover_probability":"0",
"two_points_crossover_probability":"0.9","num_of_folds":"5"}', NULL, 4);

"""
