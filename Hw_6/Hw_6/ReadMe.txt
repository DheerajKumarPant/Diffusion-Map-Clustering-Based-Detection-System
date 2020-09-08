There is single python files in submission, 'analysis.py'.

load_cached_matrices function is used to store precalculated matrices by using pickle,

To load data during training datahandler function is provided in analysis.py that is called in line 205 in analysis.py, "to input data just give the name of file containing data". 

Number of components is set to 3 by default , for plotting purpose 
Value of sigma is set to 5

In analysis.py execution begins from " if __name__ == '__main__': "
First data is loaded which is to be analysed.


data_handler() function take filename of data as argument and returns data matrix and corresponding labels which are chosen randomly.
calculate_matrices function is used to calculate major intermediate matrices and gave us two options:
1. Use the previously calculated cached data to do further calculations and plot results. 
2. Generate new cached data and over-write it to previously calculated one, and then do further calculations and plot results.
For 1. we have to set cached = true and dump = false in line no 207 where calculate_matrices function is called. For dumping 
data set cached=false and dump=true