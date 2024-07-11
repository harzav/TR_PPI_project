import configparser
import json
import os
import random
import sys
import time
import logging
import dataset_preprocessing as dp
import psycopg2
import datetime
from shutil import copy


def start_next_job():
    # Init configuration parser
    conn, config, pid = config_connect_db()
    # check if there are running jobs
    newjob = find_next_job(conn, config, pid)
    # Set job running
    set_job_running(newjob[0], conn, newjob[1], pid)

    # get jobs details
    jobid = newjob[0]
    user = newjob[1]

    # parse job parameters
    jobparams = json.loads(newjob[2])

    # run process job
    result = dp.preprocess_data(jobparams['input_dataset'], jobparams['variables_for_normalization_string'],
                                jobparams['output_folder'], "preprocessed_data_{}.txt".format(jobid),
                                float(jobparams['missing_threshold']), int(jobparams['missing_imputation_method']),
                                int(jobparams['normalization_method']), int(jobparams['filetype']),
                                int(jobparams['has_features_header']), int(jobparams['has_samples_header']), user,
                                jobid, pid)

    # # check if everything went OK
    if result[0] == 0:
        # Encountered an error
        set_job_error(jobid, conn, user, result={'error': result[1]}, pid=pid)
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Preprocessing Data process has finished "
                     "unsuccessfully".format(pid, jobid, user))
        sys.exit("Biomarkers Preprocessing Data process has finished unsuccessfully")

    else:

        # move results to user's DS
        timestamp = int(time.time())
        filerand = random.randint(1000, 9999)
        jobuser_list = user.split("@")
        datastore_of_user_path = "{}{}_at_{}/".format(config['insybio.datastore']['datastore'], jobuser_list[0],
                                                      jobuser_list[1])
        newfilename = "{}preprocessed_data{}_{}.txt".format(datastore_of_user_path, timestamp, filerand)
        orig_filename = "{}preprocessed_data_{}.txt".format(jobparams['output_folder'], jobid)
        copy(orig_filename, newfilename)

        description = "Preprocessed file from {}".format(jobparams['dataset_title'])
        dsfileID = update_ds(newfilename, orig_filename, conn, description, int(jobparams['filetype']), 2, user, jobid,
                             pid)
        message = {"output": os.path.basename(orig_filename), "dsfileid": dsfileID}

        # Everything successful
        set_job_completed(jobid, conn, message, user, pid)
        logging.info(
            "PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Preprocessing Data process has "
            "finished successfully".format(pid, jobid, user))
        sys.exit("Biomarkers Preprocessing Data process has finished successfully")


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
    Find next job with status 1 and type 2 = Preprocessing
    :param conn: db connection object
    :param pid: job's pid
    :param config: configuration dictionary
    :return: newjob: list with job_id, job_user and job's input file path
    """
    # define cursor
    cur = conn.cursor()

    # check if there are running jobs
    query = "SELECT count(id) as model_training from biomarkers_jobs WHERE status=2 AND type=2"
    cur.execute(query)
    numRunningJobs = cur.fetchone()
    # Check if number of running jobs is grater that what the ini specifies uncomment in production
    if numRunningJobs[0] >= int(config['biomarkers.preprocessing']['paralleljobs']):
        logging.info("There are still running jobs")
        sys.exit()

    # get next job to run
    try:
        cur.execute("""SELECT id,\"user\",input from biomarkers_jobs WHERE status=1 AND type=2 ORDER BY
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


def update_ds(filepath, orig_filepath, conn, description, filetype, fileformat, job_user='unknown', jobid=0, pid=0):
    """
    Inserts the new biological file in the datastore
    :param filepath: the absolute path to the new file
    :param orig_filepath: the absolute path to the file's original name
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
    filename = os.path.basename(orig_filepath)
    targetfile = os.path.basename(filepath)
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


if __name__ == "__main__":
    start_next_job()

# For testing purposes insert the following data after changing the input and output file path:

# INSERT INTO public.biomarkers_jobs VALUES (4, 'info@insybio.com', 13410000, NULL, NULL, 1,
# '{"input_dataset":"\/opt\/backend-application\/insybio-biomarkers\/02.Dataset_Preprocessing\/Input\/twoclass\/
# new_split_0.1/data_train.txt","selected_features_string":"",
# "output_folder":"\/opt\/backend-application\/insybio-biomarkers\/02.Dataset_Preprocessing\/Output\/",
# "missing_threshold":"0.05","missing_imputation_method":"1","normalization_method":"1","filetype":"18",
# "has_features_header":"0","has_samples_header":"0"}', NULL, 2);

# INSERT INTO public.biomarkers_jobs VALUES (4, 'info@insybio.com', 13410000, NULL, NULL, 1,
# '{"input_dataset":"\/opt\/backend-application\/insybio-biomarkers\/02.Dataset_Preprocessing\/Input\/twoclass\/
# new_split_0.1/data_train.txt","selected_features_string":"",
# "output_folder":"\/opt\/backend-application\/insybio-biomarkers\/02.Dataset_Preprocessing\/Output\/",
# "missing_threshold":"0.05","missing_imputation_method":"1","normalization_method":"1","filetype":"18",
# "has_features_header":"0","has_samples_header":"0"}', NULL, 2);


