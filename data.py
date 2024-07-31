import pandas as pd
import numpy as np

## always the convention: 
# N datapoints and M variables

params = ['FRC','Dose','Conductivity','Temperature','pH','DO','Turbidity','ORP']

# Sets of files to use in analysis
all_files = ['Distilled', 'Tap', 'MC_1', 'MC_2', 'Pond', 'Pool', 'JL', 'Kenya_Ivonangya_1', 'Kenya_jerry_cans', 'Kenya_Ivonangya_2', 'Kenya_Mumo_1', 'Kenya_Mumo_2', 'Kenya_Kyuso', 'Kenya_Hostel']
Kenya_files = ['Kenya_Ivonangya_1', 'Kenya_jerry_cans', 'Kenya_Ivonangya_2', 'Kenya_Mumo_1', 'Kenya_Mumo_2', 'Kenya_Kyuso', 'Kenya_Hostel']
cam_files = ['Distilled', 'Tap', 'MC_1', 'MC_2', 'Pond', 'Pool', 'JL']
low_conductivity_files = ['Distilled_scaled', 'Tap_scaled']
DO_files = ['Distilled', 'Tap', 'MC_1', 'MC_2', 'Pond', 'Pool', 'JL', 'Kenya_Ivonangya_2', 'Kenya_Mumo_2', 'Kenya_Hostel']
non_pool_files = ['Distilled', 'Tap', 'MC_1', 'MC_2', 'Pond', 'JL', 'Kenya_Ivonangya_1', 'Kenya_jerry_cans', 'Kenya_Ivonangya_2', 'Kenya_Mumo_1', 'Kenya_Mumo_2', 'Kenya_Kyuso', 'Kenya_Hostel']

def fetch_data(included_sets):
    '''Function to fetch dictionary of datasets.
    inputs
        included_sets:  list of strings     sets of data to include
    
    returns a dictionary of datasets {string: DataFrame}'''

    datasets = {}
    for set in included_sets:
        data = pd.read_csv(r'Data/{}.csv'.format(set))
        datasets[set] = data
    return datasets

def normalise(x, return_params = False):
    '''Normalise data for each variable. Calculate the mean and standard deviation 
    and scale the data to set the mean to 0 and standard deviation to 1.
    
    Inputs:
        x               (N x M) array   N datapoints and M variables
        return_params   bool            return if we want to maintain the mean and sd.
    
    Returns the normalised x (and mean and sd).'''

    num_datapoints, _ = x.shape # obtain the number of rows and columns

    m = x.sum(axis=0)/num_datapoints     # compute the mean along each column and collect into a vector
    x0 = x - m[np.newaxis,:] # subtract the mean from each element
                            # the "np.newaxis" construct creates identical rows from the same mean value

    sd = np.sqrt((x0**2).sum(axis=0)/num_datapoints) # now compute the standard deviation of each column
    ss = np.array([tmp if tmp != 0 else 1 for tmp in sd]) # if the standard deviation is zero, replace it with 1
                                                       # to avoid division by zero error
    x00 = x0 / ss[np.newaxis,:]    # divide each element by the corresponding standard deviation
    if return_params:
        return x00,m,ss
    return x00     # return the normalised data matrix

def scaling(x,m,ss):
    '''Scale input data by given mean and standard deviation to normalise it.
    
    Inputs:
        x       (N x M) array   N datapoints and M variables
    Returns the changed x.'''
    x0 = x - m[np.newaxis,:] # subtract the mean from each element
    x00 = x0 / ss[np.newaxis,:]    # divide each element by the corresponding standard deviation
    return x00                     # return the normalised data matrix

def plot_2d(datasets, x_var, y_var, ax):
    '''Create a scatter plot of datapoints from the chosen datasets for the chosen 2 variables.

    Inputs
        datasets:   dict            all datasets in use
        x_var:      string          parameter along x axis
        y_var:      string          parameter along y axis
        ax:         axis object     object from matplotlib
    
    Return an (M x 2) array of all the points plotted.
    '''

    all_x = []
    all_y = []
    for set in datasets:
        df = datasets[set][datasets[set][x_var] > -1]
        df = df[df[y_var] > -1]
        x = df[x_var].tolist()
        y = df[y_var].tolist()

        for i in range(len(x)):
            all_x.append(x[i])
            all_y.append(y[i])
        
        if set == 'DW':
            set = 'Distilled'
        ax.scatter(x,y, label = set)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    return all_x, all_y

def form_dataframe(parameters, chosen_sets, range_var, lb, ub):
    '''Create a formatted dataframe with datapoints that include all required fields
    and in the right FRC range.
    
    Inputs
        parameters:     [str]   variables to include
        chosen_sets:    [str]   datasets to include
        range_var:      str     variable over which we are constricting to a specified range
        lb:             float   lower bound of specified range
        up:             float   upper bound of specified range
    
    Return a dataframe of complete entries for the specified parameters.
    '''

    dimension = len(parameters)
    datasets = fetch_data(chosen_sets)

    df = pd.DataFrame()

    for set in datasets:
        new = datasets[set][datasets[set][parameters[0]] > -1]
        for i in range(1,dimension):
            new = new[new[parameters[i]] > -1]
        new = new[new['FRC'] >= 0]
        new = new[new['FRC'] <= 2.5]
        new = new[new[range_var] >= lb]
        new = new[new[range_var] <= ub]
        df = pd.concat([df, new])
    
    exclude = excluded_vars(parameters)
    df = df.drop(exclude, axis = 1)
    return df

def excluded_vars(parameters):
    '''form a list of variables to exclude based on the parameters chosen.'''

    drop_vars = params.copy()
    drop_vars.remove('FRC')
    for param in parameters:
        drop_vars.remove(param)
    
    return drop_vars

def eig_decomposition(evals, evecs, X0, X, printing = True):
    '''Print eigenvectors with corresponding eigenvalues along with
    a cross correlation of the eigenvector's direction with the FRC.
    
    Inputs
        evals:          (N x 1) array   Vector of eigenvalues in descending order
        evecs:          (N x N) array   Array of corresponding eigenvectors (each eigenvector is a column)
        X0:             (M' x N) array  Normalised data array including FRC column
        X:              (M x N) array   Normalised data array with variables involved in decomposition
    
    '''
    FRC = X0[:, 0]
    
    correlations = []
    for i in range(len(evals)):
        evec_sc = evecs[:,i]
        X_new = np.matmul(X, evec_sc)
        corr = np.corrcoef(FRC, X_new)[0,1]
        correlations.append(corr)
        if printing:
            print('\nEigenvector {}: {}'.format(i+1, evec_sc.real))
            print('Eigenvalue {:}: {:.2e}'.format(i+1, evals[i].real))
            print('Correlation with FRC: {}'.format(corr))
    
    correlations = list(np.absolute(correlations))
    best_val = max(correlations)
    best_index = correlations.index(best_val)
    return best_index + 1, best_val


def form_data_LDA(chosen_sets, dimension, parameters, FRC_groups):
    '''Fetch the data from files and format it to be processed in LDA analysis.

    Inputs:
        chosen_sets:    [str]       a list of the water sources to include in this run.
        dimension:      int         the number of variables we are including
        parameters:     [str]       the variables to include
        FRC_groups:     [[float]]   a list of length-2 vectors containing the lower and 
                                    upper bounds of each 'class' we are defining
    
    Returns:
        X_prime:    (N x M') array      unnormalised array including the FRC column
        X:          (N x M) array       N = number of datapoints, M = number of variables.
        y:          (M x 1) array       labels for each datapoint
        FRC:        (M x 1) array       FRC of each datapoint
    
    '''
    datasets = fetch_data(chosen_sets)

    df = pd.DataFrame()
    for set in datasets:
        new = datasets[set][datasets[set][parameters[0]] > -1]
        for i in range(1,dimension):
            new = new[new[parameters[i]] > -1]
        df = pd.concat([df, new])
    
    drop_vars = excluded_vars(parameters)
    df = df.drop(drop_vars, axis = 1)

    X_prime = df.to_numpy()

    X = X_prime[:, 1:]
    FRC = X_prime[:, 0].flatten()
    y = FRC.copy()

    for index, group in enumerate(FRC_groups):
        bottom = group[0]
        top = group[1]
        np.place(y, y<top, index+10)
    y -= 10

    return X_prime, X, y, FRC