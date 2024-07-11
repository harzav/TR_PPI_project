'''
Strips the extracted preprocessed file from the proteins and sample tags.

Example run:
    python3 stripper.py input_filename.txt output_folder/ output_filename.txt
'''

import sys
import csv
import numpy as  np

def parse_data(data_filename):
	'''
	Parses data.

	Args:
		data_filename: dataset filename

	Returns: a list of three lists, [proteins, data, samples].
	'''
	num_of_lines=0
	proteins=list()
	data=list()
	samples=list()
	with open(data_filename) as data_fname:
		for line in csv.reader(data_fname, delimiter="\t"):
			if num_of_lines==0:
				for j in range(len(line)):
					if j>0:
						samples.append(line[j].strip())
			else:
				proteins.append(line[0])
				data.append([])
				for j in range(len(line)):
					if j>0:
						#if line[j]!='':
						try:
							data[num_of_lines-1].append(float(line[j]))						
						#else:
						except:
							data[num_of_lines-1].append('')
			num_of_lines+=1
	#print('Data were successfully parsed!')
	return [proteins,data,samples]

def print_data(data, folder_name, filename):
    '''
    Prints data.

    Args:
        data: input data
        folder_name: output folder name
        filename: output filename

    '''
    #data = np.transpose(data)
    file = open(folder_name + filename, 'w')
    message = ''
    for i in range(len(data)):
        for j in range(len(data[0])):
            message += str(data[i][j])
            if j < len(data[0]) - 1:
            	message += '\t'
        message += '\n'
    file.write(message)
    file.close()


def main():
    input_filename = sys.argv[1]
    output_folder = sys.argv[2]
    output_filename = sys.argv[3]
    proteins, data, samples = parse_data(input_filename)
    print_data(data, output_folder, output_filename)

if __name__ == "__main__":
	main()
