import configparser
import json
import os
import random
import sys
import time
import logging
import biomarkers_dataset_statistical_analysis as bdsa
# import biomarkers_dataset_statistical_analysis_new as bdsa_new
import psycopg2
import datetime
import shutil


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
    logging.basicConfig(filename="{}biomarkers_statistical_reports_{}.log".format(logs_path, todaystr),
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
    Find next job with status 1 and type 3 = Statistical Analysis
    :param conn: db connection object
    :param pid: job's pid
    :param config: configuration dictionary
    :return: newjob: list with job_id, job_user and job's input file path
    """
    # define cursor
    cur = conn.cursor()

    # check if there are running jobs
    query = "SELECT count(id) as stat_analysis_jobs from biomarkers_jobs WHERE status=2 AND type=3"
    cur.execute(query)
    numRunningJobs = cur.fetchone()
    # Check if number of running jobs is grater that what the ini specifies uncomment in production
    if numRunningJobs[0] >= int(config['biomarkers.statanalysis']['paralleljobs']):
        # logging.info("There are still running jobs")
        sys.exit()

    # get next job to run
    try:
        cur.execute("""SELECT id,\"user\",input from biomarkers_jobs WHERE status=1 AND type=3 ORDER BY
         starttimestamp ASC LIMIT 1""")  # selecting columns id, user, input
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


def check_image_parameters(parameters, pid=0, jobid=0, user='unknown'):
    """
    Check if all image parameters are present, if not fill with default values
    :param parameters: this job's input parameters
    :param pid: this job's PID
    :param jobid: this job's ID
    :param user: this job's username
    :return: filled parameters with default values if needed
    """
    try:
        # A numeric input (size of volcano png width in cm)
        if "volcano_width" not in parameters:
            parameters["volcano_width"] = 12
        # A numeric input (size of volcano png height in cm)
        if "volcano_height" not in parameters:
            parameters["volcano_height"] = 8
        # A numeric input  (size of all volcano titles)
        if "volcano_titles" not in parameters:
            parameters["volcano_titles"] = 8
        # A numeric input  (size of axis titles)
        if "volcano_axis_labels" not in parameters:
            parameters["volcano_axis_labels"] = 8
        # A numeric input (size of significant genes labels names)
        if "volcano_labels" not in parameters:
            parameters["volcano_labels"] = 2
        # A numeric input (the relevance between the 2 axis size. 3/4 could be the the default because it fills
        # the whole png)
        if "volcano_axis_relevance" not in parameters:
            parameters["volcano_axis_relevance"] = 3/4
        # A numeric input (between 1,2 and 3)
        if "volcano_criteria" not in parameters:
            parameters["volcano_criteria"] = 3
        # A numeric (float) number between (0 and 1)
        if "abs_log_fold_changes_threshold" not in parameters:
            parameters["abs_log_fold_changes_threshold"] = 0
        # A numeric input (between 0,1 and 2)
        if "volcano_labeled" not in parameters:
            parameters["volcano_labeled"] = 2

        # A numeric input (size of heatmap png width in cm)
        if "heatmap_width" not in parameters:
            parameters["heatmap_width"] = 15
        # A numeric input (size of heatmap png height in cm)
        if "heatmap_height" not in parameters:
            parameters["heatmap_height"] = 15
        # An option between "hierarchical" and "none"
        if "features_hier" not in parameters:
            parameters["features_hier"] = "hierarchical"
        # An option between "euclidean","manhattan","maximum"
        if "features_metric" not in parameters:
            parameters["features_metric"] = "euclidean"
        # An option between "average","single","complete","centroid","ward.D"
        if "features_linkage" not in parameters:
            parameters["features_linkage"] = "complete"
        # An option between "hierarchical" and "none"
        if "samples_hier" not in parameters:
            parameters["samples_hier"] = "hierarchical"
        # An option between "euclidean","manhattan","maximum"
        if "samples_metric" not in parameters:
            parameters["samples_metric"] = "euclidean"
        # An option between "average","single","complete","centroid","ward.D"
        if "samples_linkage" not in parameters:
            parameters["samples_linkage"] = "single"
        # An option between 1 and 0 (to show the z-score bar or not)
        if "heatmap_zscore_bar" not in parameters:
            parameters["heatmap_zscore_bar"] = 1

        # A numeric input (size of beanplot png width in cm)
        if "beanplot_width" not in parameters:
            parameters["beanplot_width"] = 20
        # A numeric input (size of beanplot png height in cm)
        if "beanplot_height" not in parameters:
            parameters["beanplot_height"] = 20
        # A numeric input ( scaling of axis (scaling relative to default, e.x. 1=default, 1.5 is 50% larger, 0.5 is 50%
        # smaller) )
        if "beanplot_axis" not in parameters:
            parameters["beanplot_axis"] = 1.6
        # A numeric input ( scaling of axis x (scaling relative to default, e.x. 1=default, 1.5 is 50% larger, 0.5 is
        # 50% smaller) )
        if "beanplot_xaxis" not in parameters:
            parameters["beanplot_xaxis"] = 1.3
        # A numeric input ( scaling of axis y (scaling relative to default, e.x. 1=default, 1.5 is 50% larger, 0.5 is
        # 50% smaller) )
        if "beanplot_yaxis" not in parameters:
            parameters["beanplot_yaxis"] = 1.3
        # A numeric input ( scaling of titles (scaling relative to default, e.x. 1=default, 1.5 is 50% larger, 0.5 is
        # 50% smaller) )
        if "beanplot_titles" not in parameters:
            parameters["beanplot_titles"] = 1.6
        # A numeric input ( scaling of axis titles (scaling relative to default, e.x. 1=default, 1.5 is 50% larger,
        # 0.5 is 50% smaller) )
        if "beanplot_axis_titles" not in parameters:
            parameters["beanplot_axis_titles"] = 1.6
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during parsing default image"
                          " parameters".format(pid, jobid, user))
    return parameters


def start_next_job():
    """
    Searched db for the next available job in the queue and perform Biomarkers Training Multibiomarker Predictive
    Analytics Model
    :return: nothing just end the script
    """

    # Init configuration parser
    conn, config, pid = config_connect_db()
    # Find next job to calculate
    newjob = find_next_job(conn, config, pid)
    # Set job running
    set_job_running(newjob[0], conn, newjob[1], pid)
    # Process current job and end accordingly

    # get jobs details
    jobid = newjob[0]
    jobuser = newjob[1]
    # parse job parameters
    jobparams = json.loads(newjob[2])
    jobparams = check_image_parameters(jobparams, pid, jobid, jobuser)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tJob Retrieved with params: {} ".format(pid, jobid, jobuser, jobparams))
    output_dictionary = {}

    try:
        sys.setrecursionlimit(15000)

        result = bdsa.meta_statistical_analysis(
            jobparams['biomarkers_dataset'], jobparams['labels_filename'], jobparams['selected_comorbidities_string'],
            jobparams['output_folder_name'], int(jobparams['filetype']), int(jobparams['has_features_header']),
            int(jobparams['has_samples_header']), int(jobparams['paired_flag']), int(jobparams['logged_flag']),
            float(jobparams['pvalue_threshold']), jobparams['parametric_flag'], int(jobparams['volcano_width']),
            int(jobparams["volcano_height"]), int(jobparams["volcano_titles"]), int(jobparams["volcano_axis_labels"]),
            int(jobparams["volcano_labels"]), float(jobparams["volcano_axis_relevance"]),
            int(jobparams["volcano_criteria"]), float(jobparams["abs_log_fold_changes_threshold"]),
            int(jobparams["volcano_labeled"]), int(jobparams["heatmap_width"]), int(jobparams["heatmap_height"]),
            jobparams["features_hier"], jobparams["features_metric"], jobparams["features_linkage"],
            jobparams["samples_hier"], jobparams["samples_metric"], jobparams["samples_linkage"],
            int(jobparams["heatmap_zscore_bar"]), int(jobparams["beanplot_width"]), int(jobparams["beanplot_height"]),
            float(jobparams["beanplot_axis"]), float(jobparams["beanplot_xaxis"]), float(jobparams["beanplot_yaxis"]),
            float(jobparams["beanplot_titles"]), float(jobparams["beanplot_axis_titles"]), jobuser, jobid, pid)

        logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis done!".format(pid, jobid, jobuser))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during Statistical Analysis".format(
            pid, jobid, jobuser))
        output_dictionary['error'] = "An exception occurred during Statistical Analysis: {}".format(e)
        # return [True, output_dictionary]
        set_job_error(jobid, conn, jobuser, result=output_dictionary, pid=pid)
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Statistical Analysis process has finished "
                     "unsuccessfully".format(pid, jobid, jobuser))
        sys.exit("Biomarkers Statistical Analysis process has finished unsuccessfully")

    if result[0] == 0:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during Statistical Analysis: {}".format(
            pid, jobid, jobuser, str(result[1])))
        output_dictionary['error'] = "An exception occurred during Statistical Analysis: {}".format(str(result[1]))
        # return [True, output_dictionary]
        set_job_error(jobid, conn, jobuser, result=output_dictionary, pid=pid)
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Statistical Analysis process has finished "
                     "unsuccessfully".format(pid, jobid, jobuser))
        sys.exit("Biomarkers Statistical Analysis process has finished unsuccessfully")
    else:
        logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStatistical Analysis: {}".format(pid, jobid, jobuser, str(result[1])))
        logging.debug("Finished run number: ", result[2])
        if result[2] < 2:
            output_directory = jobparams['output_folder_name']
            try:
                statistical_analysis_outfiles = store_mq_files_in_datastore(
                    jobparams, config['insybio.datastore']['datastore'], conn, jobuser, jobid, pid)
            except Exception:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during Statistical Analysis".format(
                    pid, jobid, jobuser))
                output_dictionary['error'] = "An exception occurred during Statistical Analysis: {}".format(str(result[1]))
                # return [True, output_dictionary]
                set_job_error(jobid, conn, jobuser, result=output_dictionary, pid=pid)
                sys.exit("Biomarkers Statistical Analysis process has finished unsuccessfully")
            statistical_analysis_outfiles["multilabel"] = 0
            set_job_completed(jobid, conn, statistical_analysis_outfiles, jobuser, pid)
            logging.info(
                "PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Statistical Analysis process has finished successfully".format(
                    pid, jobid, jobuser))
        else:
            output_directory = jobparams['output_folder_name']
            multilabel_statistical_analysis_outfiles = {}
            for i in range(result[2]):
                logging.info("PID:{}\tJOB:{}\tUSER:{}\tStatistical Analysis storing Label list {}".format(pid, jobid,
                                                                                                           jobuser, i))
                jobparams['output_folder_name'] = output_directory + "Output_" + str(i) + "/"
                try:
                    statistical_analysis_outfiles = store_mq_files_in_datastore(
                        jobparams, config['insybio.datastore']['datastore'], conn, jobuser, jobid, pid)
                except Exception:
                    logging.exception(
                        "PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during Statistical Analysis".format(
                            pid, jobid, jobuser))
                    output_dictionary['error'] = "An exception occurred during Statistical Analysis: {}".format(
                        str(result[1]))
                    # return [True, output_dictionary]
                    set_job_error(jobid, conn, jobuser, result=output_dictionary, pid=pid)
                    sys.exit("Biomarkers Statistical Analysis process has finished unsuccessfully")
                multilabel_statistical_analysis_outfiles["Output_" + str(i)] = statistical_analysis_outfiles
            multilabel_statistical_analysis_outfiles["multilabel"] = 1
            set_job_completed(jobid, conn, multilabel_statistical_analysis_outfiles, jobuser, pid)
            logging.info(
                "PID:{}\tJOB:{}\tUSER:{}\tBiomarkers Statistical Analysis process has finished successfully".format(
                    pid, jobid, jobuser))
        compress_all_outputfiles(output_directory, jobuser, jobid, pid)
        sys.exit("Biomarkers Statistical Analysis process has finished successfully")


def compress_all_outputfiles(output_directory, jobuser, jobid, pid):
    """
    Create compressed directory of all produced files
    :param output_directory: directory with output files
    :param jobuser:this job's user
    :param jobid: this job's id
    :param pid:this job's PID
    :return:
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tompressing Output Directory".format(pid, jobid, jobuser))
    try:
        output_directory_archive = output_directory.strip('/')
        shutil.make_archive(output_directory_archive, 'zip', output_directory)
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception occurred during compressing output directory "
                          "Statistical Analysis".format(pid, jobid, jobuser))


def store_mq_files_in_datastore(parameters, datastore_path, conn, user, jobid, pid):
    """
    Store Molecule quantification files produced from step 4 into the datastore
    :param parameters: runtime parameters
    :param datastore_path: user's datastore path
    :param conn: database connection object
    :param user: this job's user
    :param jobid: this job's id
    :param pid:this job's PID
    :return: dictionary with produced files
    """
    output_dictionary = {}
    jobuser_list = user.split("@")
    datastore_of_user_path = '{}{}_at_{}/'.format(datastore_path, jobuser_list[0], jobuser_list[1])

    # Storing names and filesizes
    mq_files = os.listdir(parameters['output_folder_name'] + 'MQ_files/')
    logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStoring file {}".format(pid, jobid, user, mq_files))
    mq_files_ids = []
    mq_files_paths = []
    for mq in mq_files:
        description = "MQ file ({}) created from ({}) and ({})".format(mq, parameters['dataset_title'],
                                                                       parameters['labels_title'])
        logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStoring file: {}".format(pid, jobid, user, description))
        datastoreid, _ = copy_file_from_output_to_user_datastore(
            mq, parameters['output_folder_name'] + "MQ_files/", datastore_of_user_path, conn, description, 7, 2, user,
            jobid, pid)
        mq_files_ids.append(datastoreid)
        mq_files_paths.append('{}MQ_files/{}'.format(parameters['output_folder_name'], mq))

    sign_molecules_file = 'significant_molecules_dataset.tsv'
    sign_molecules_file_path = '{}{}'.format(parameters['output_folder_name'], sign_molecules_file)
    if os.path.isfile(sign_molecules_file_path):

        description = "Significant molecules dataset ({}) created from ({}) and ({})".format(
            sign_molecules_file, parameters['dataset_title'], parameters['labels_title'])
        logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStoring file: {}".format(pid, jobid, user, description))
        datastoreid, _ = copy_file_from_output_to_user_datastore(
            sign_molecules_file, parameters['output_folder_name'],
            datastore_of_user_path, conn, description, 19, 2, user, jobid, pid)
        molecules_file_id = datastoreid

        mq_sign_files = os.listdir(parameters['output_folder_name'] + 'MQ_significant_files/')
        mq_significant_files_ids = []
        mq_sign_files_paths = []
        for mq_significant in mq_sign_files:
            description = "MQ significant file ({}) created from ({}) and ({})".format(
                mq_significant, parameters['dataset_title'], parameters['labels_title'])
            logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStoring file: {}".format(pid, jobid, user, description))
            datastoreid, _ = copy_file_from_output_to_user_datastore(
                mq_significant, '{}MQ_significant_files/'.format(parameters['output_folder_name']),
                datastore_of_user_path, conn, description, 7, 2, user, jobid, pid)
            mq_significant_files_ids.append(datastoreid)
            mq_sign_files_paths.append('{}MQ_significant_files/{}'.format(parameters['output_folder_name'],
                                                                          mq_significant))

        diff_exp_files = os.listdir(parameters['output_folder_name'] + "diff_expression_files/")
        diff_exp_files_filesizes = [os.path.getsize('{}diff_expression_files/{}'.format(
            parameters['output_folder_name'], file)) for file in diff_exp_files]
        diff_exp_files_nonzero = [diff_exp_files[i] for i in range(len(diff_exp_files))
                                  if diff_exp_files_filesizes[i] > 0]
        de_files_ids = []
        diff_exp_files_paths = []
        for diff_exp in diff_exp_files_nonzero:
            description = "Differential expression file created from {}".format(diff_exp[18:-4])
            logging.debug("PID:{}\tJOB:{}\tUSER:{}\tStoring file: {}".format(pid, jobid, user, description))
            datastoreid, _ = copy_file_from_output_to_user_datastore(
                diff_exp, '{}diff_expression_files/'.format(parameters['output_folder_name']),
                datastore_of_user_path, conn, description, 8, 2, user, jobid, pid)
            de_files_ids.append(datastoreid)
            diff_exp_files_paths.append('{}diff_expression_files/{}'.format(parameters['output_folder_name'], diff_exp))

        output_dictionary = \
            {"MQ_file_IDs": mq_files_ids,
             "MQ_original_names": mq_files,
             "MQ_sign_file_IDs": mq_significant_files_ids,
             "MQ_sign_original_names": mq_sign_files,
             "sign_molecules_ID": molecules_file_id,
             "sign_molecules_original_name": sign_molecules_file,
             "differential_expression_file_IDs": de_files_ids,
             "differential_expression_file": diff_exp_files_nonzero}
        return output_dictionary
    else:
        output_dictionary = {"MQ_file_IDs": mq_files_ids, "MQ_original_names": mq_files}
        return output_dictionary


def copy_file_from_output_to_user_datastore(original_file, directory, datastore_of_user_path, conn, description,
                                            file_type, file_format, user, jobid, pid):
    """
    Copy file from working path to user's datastore folder
    :param original_file: file to move to datastore
    :param directory: working path directory
    :param datastore_of_user_path: user's datastore path
    :param conn: Database connection object
    :param description: job's description for datastore file storage
    :param file_type: this files type
    :param file_format: this files format
    :param user: this job's user
    :param jobid: this job's ID
    :param pid:this job's PID
    :return: datastore ID and file's name that moved to datastore folder
    """
    timestamp = int(time.time())
    if file_format == 2:
        newfilename = 'dsfile{}_{}.tsv'.format(timestamp, random.randint(1000, 9999))
    else:
        newfilename = 'dsmodelsfile{}_{}.zip'.format(timestamp, random.randint(1000, 9999))
    shutil.copyfile('{}{}'.format(directory, original_file), '{}{}'.format(datastore_of_user_path, newfilename))
    datastoreid = update_local_ds('{}{}'.format(datastore_of_user_path, newfilename), original_file, conn, description,
                                  file_type, file_format, user, jobid, pid)

    return datastoreid, newfilename


def update_local_ds(filepath, original_name, conn, description, filetype, fileformat, job_user='unknown', jobid=0,
                    pid=0):
    """
    Inserts the new biological file in the datastore with a given targetfilepath and original_name
    :param filepath: the absolute path to the new file
    :param original_name: the original name of the file
    :param conn: db connection object
    :param description: new file's description
    :param filetype: file's type according to ds_filetypes table
    :param fileformat: file's format according to ds_fileformats table
    :param job_user: job's user id
    :param jobid: job's id
    :param pid: this job's Pid
    :return: file's in datastore new entry id
    """

    timestamp = int(time.time())
    # filepath = Path(file).resolve()
    filesize = os.path.getsize(filepath)
    filename = os.path.basename(original_name)
    targetfile = os.path.basename(filepath)
    # targetfile = "/".join((os.path.basename(os.path.dirname(filepath)), filename))
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
        cur.execute(insert_query, (jobid, insertid, fileformat))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in DB insertion of the new created biological file.".format(
            pid, jobid, job_user))
        return False

    conn.commit()
    return insertid


if __name__ == "__main__":
    start_next_job()
