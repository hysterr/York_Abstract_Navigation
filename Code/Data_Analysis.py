import pandas as pd 
import pickle 
import numpy as np 
import sys

'''
	This is an interactive python script for data analysis. It should be run from the command line
	in interactive mode. To enter interactive mode, use the following command:

		python -i Data_Analysis.py

	Once in interactive mode, all of the fuctions can be used and called by the user without 
	requiring the script to be re-run. 

	If the script is modified, it will have to run again. To exit interactive mode, press CTRL+D 
	and this will return the user to the command line. 

	This script can only read data which is stored in the correct file system. See example file structure

	- Framework
		- Simulation Data
			- Test 1
			- Test 2
			- Test X

		- Code 
			- Data_Analysis.py
			- Coop_Task.py
			- Utilities 
				- Python files for simulation

	This script Data_Analysis.py is located inside the /Code folder of the Framework directory. The data 
	failes for analysis should be one folder up inside /Simulation Data with each test labelled as 
	'/Test {test_number}'
'''



# =============================================================================
# Analyse Simulation Result
# =============================================================================
def Load_Simulation(test_number, max_number=None):
	file_path = f'../Simulation Data/Test {test_number}/Results.csv'
	try:

		loaded_results = np.genfromtxt(f'../Simulation Data/Test {test_number}/Results.csv', delimiter=',')

		# Allow the user to use only a subset of the data, provided the data exists.
		if max_number is not None:
			try:
				loaded_results = loaded_results[:max_number]
			except:
				print("Not enough data points for 'max_number' value entered.")

		total = loaded_results.shape[0]
		success = loaded_results.sum(0)[1]

		print("+ Total of %.0f resuls loaded with %.0f success values (%.2f%%)" % (total, success, (success/total)*100))

		failed_episodes = list()
		for value in loaded_results:
			if value[1] == 0:
				failed_episodes.append(value[0].astype(int))
		print("Extracted %.0f failed episodes. See variable 'failed_episodes' for more." % (len(failed_episodes)))


	except IOError:
		print("Simulation files do not exist. Tried to look inside ", file_path)
		loaded_results = None
		failed_episodes = None


	return loaded_results, failed_episodes
# =============================================================================




# =============================================================================
# Load Episode Function - Takes the test_number as the input argument from the
# command line argument and uses that to load the corresponding path. The 
# episode relates to the episode number to load. 
# 
# This function allows individual episodes from any test to be loaded. 
# =============================================================================
def Load_Episode(test_number, episode):
	# print(f"Loading episode {episode} in test {test}")
	file_path = f"../Simulation Data/Test {test_number}/"
	file_name = f"Episode_{episode}"

	d_csv = pd.read_csv(file_path+file_name+".csv", delimiter=",")

	with open(file_path+file_name+"_Data.pickle", 'rb') as pickle_file:
		d_epi = pickle.load(pickle_file)

	return d_csv, d_epi
# =============================================================================


# =============================================================================
# Analyse_Full_Simulation
# =============================================================================
def Analyse_Full_Simulation(test_number):
	# Find total number of simulations
	file_path = f'../Simulation Data/Test {test_number}/'
	loaded_results = np.genfromtxt(file_path+"Results.csv", delimiter=',')
	n_files = loaded_results.shape[0]
	print("Analysing ", n_files, "... this could take a while!")
	pd_headers = ["Episode", "Complete", "N Steps", "Success", "Return", "Fail", "Hold", "Redirect"]
	data_array = np.empty(shape=(n_files, len(pd_headers)), dtype='<U21')

	# Iterate through each episode during the simulation
	for i in range(n_files):
		file_name = f"Episode_{i+1}"
		d_csv = pd.read_csv(file_path+file_name+".csv", delimiter=",")

		# Create a subset of the dataframe where the values are a certain value, and obtain the number.
		data_array[i,:] = np.array([
			i+1,
			loaded_results[i,1].astype(int),
			d_csv.shape[0],
			d_csv[d_csv.State == 'success'].shape[0],
			d_csv[d_csv.State == 'return'].shape[0],
			d_csv[d_csv.State == 'fail'].shape[0],
			d_csv[d_csv.State == 'Hold'].shape[0],
			d_csv[d_csv.State == 'Redirect'].shape[0],
		])

	# Compile data into a dataframe for readability
	df = pd.DataFrame(data_array, columns=pd_headers)

	return df
# =============================================================================








# =============================================================================

# =============================================================================
def Analyse_Fail_State(data):
	for i, j in enumerate(data["State"]):
		if j == 'fail':
			fail_state = i

	fail_step = data[fail_state:]

	return fail_step
# =============================================================================






# =============================================================================
# Print 
#
# When an epsisode has been loaded by the load_episode() function, it will
# return a data dictionary 'd_epi'. The print function takes the dict 'd_epi' 
# as the input for 'data' and by passing a single step number will print the 
# information as a readable output.
# =============================================================================
def Print(data, step_number):
	print("Human information:")
	for h in data[step_number]['human']:
		string = str(data[step_number]['human'][h])
		print("\t%-30s %-30s" % (h, string))

	print("Agent information")
	for a in data[step_number]['agent']:
		string = str(data[step_number]['agent'][a])
		print("\t%-30s %-30s" % (a, string))
# =============================================================================




# # Load indiviual episodes for analysis 
# episode = 1


# with open('out/cache/' +hashed_url, 'rb') as pickle_file:
#     content = pickle.load(pickle_file)