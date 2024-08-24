import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import data
import matplotlib.gridspec as gridspec

colours = ['blue', 'red', 'mediumseagreen', 'magenta', 'darkorange', 'aqua', 'dodgerblue', 'darkorchid', 'indianred', 'goldenrod', 'grey', 'black', 'lightgreen', 'teal', 'mediumslateblue']

def linreg_2d(x_var, y_var, chosen_sets):
    '''Plot a line of best fit for the data chosen'''

    datasets =  data.fetch_data(chosen_sets)

    fig, ax = plt.subplots(1,1)

    x,y = data.plot_2d(datasets, x_var, y_var, ax)

    slope, intercept, r, p, se = scipy.stats.linregress(x, y)

    ax.plot(x, intercept + slope*x, 'r', label='$R^2 = {:.3f}$'.format(r**2))
    plt.legend()
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title('{} vs {}'.format(x_var, y_var))

    fig.savefig('Graphs/{}/{}.pdf'.format(y_var, 'all'), format = 'pdf', transparent=False, bbox_inches='tight')
    plt.show()

def PCA_bounded_FRC(parameters, chosen_sets, FRC_bot, FRC_top):
# def PCA_bounded_FRC(X_prime, parameters, dimension, FRC_bot, FRC_top, include_FRC = False):
    '''Conduct a PCA with datapoints bounded by an FRC range. 
    Inputs
        parameters:     [str]       variables to include
        chosen_sets:    [str]       datasets to include
        FRC_bot:        float       lower bound for the FRC range
        FRC_top:        float       upper bound for the FRC range
        include_FRC:    bool        choose whether to include the FRC in our eigenvalue decomposition.
    
    Outputs:
        X0:     (M' x N) array  Normalised data array including FRC column
        X:      (M x N) array   Normalised data array with variables involved in decomposition
        evecs:  (N x N) array   Array of corresponding eigenvectors (each eigenvector is a column)
    '''

    dimension = len(parameters)
    range_var = 'FRC'
    df = data.form_dataframe(parameters, chosen_sets, range_var, FRC_bot, FRC_top)
    dimension = dimension
    X_prime = df.to_numpy()
    X0 = data.normalise(X_prime)
    X = data.normalise(X_prime[:,1:])
    XTX = np.matmul(np.transpose(X), X) 

    evals,evecs = np.linalg.eigh(XTX)
    evals = np.flip(evals)
    evecs = np.flip(evecs, axis = 1)

    best_PC, best_val = data.eig_decomposition(evals, evecs, X0, X)

    gs = gridspec.GridSpec(1, dimension)
    fig = plt.figure(figsize=(5*dimension, 5), dpi=80)

    pc_sets = [[0,1], [0,2], [0,3], [0,4]]

    col = 0
    for i in range(dimension-1):
        pc1 = pc_sets[i][0]
        pc2 = pc_sets[i][1]

        X1 = np.matmul(X,evecs[:,pc1])
        X2 = np.matmul(X,evecs[:,pc2])
        colouring = X_prime[:,0]

        ax = fig.add_subplot(gs[0, i])
        col = ax.scatter(X1, X2,c=colouring, cmap="GnBu")
        ax.set_xlabel('PC{}'.format(pc1+1))
        ax.set_ylabel('PC{}'.format(pc2+1))
        ax.set_title('PC{} vs PC{}'.format(pc1+1, pc2+1))

    plt.colorbar(col, label = 'FRC')
    plt.show()
    return X0, X, evecs, best_PC, X_prime, best_val

def PCA_for_scoring(X_prime):
    '''Conduct PCA on the provided dataset. This function is primarily used for many runs to obtain
    an average score.

    Inputs
        X_prime:    (M' x N) array     unnormalised data including FRC column.

    Outputs
        X0:         (M' x N) array      Normalised data array including FRC column
        X:          (M x N) array       Normalised data array with variables involved in decomposition
        evecs:      (N x N) array       Array of corresponding eigenvectors (each eigenvector is a column)
        best_PC     int                 the number of the PC best aligned with the FRC
        best_val    float               correlation of that PC with the FRC
    '''

    X0 = data.normalise(X_prime)

    X = data.normalise(X_prime[:,1:])
    XTX = np.matmul(np.transpose(X), X) 

    evals,evecs = np.linalg.eigh(XTX)
    evals = np.flip(evals)
    evecs = np.flip(evecs, axis = 1)

    best_PC, best_val = data.eig_decomposition(evals, evecs, X0, X, printing=False)
    return X0, X, evecs, best_PC, best_val

def calc_range(x):
    minimum = min(x)
    scale = max(x) - min(x)
    return minimum, scale

def calc_score_three_groups(X, evecs, best_PC, X_prime):
    '''Function to calculate the proportion of correctly classified points for a split of 3 groups.
    
    Inputs
        X:          (M x N) array       Normalised data array with variables involved in decomposition
        evecs:      (N x N) array       Array of corresponding eigenvectors (each eigenvector is a column)
        best_PC     int                 the number of the PC best aligned with the FRC
        X_prime:    (M' x N) array      unnormalised data including FRC column.

    Outputs
        score:      float               proportion of correctly classified points.
        PC:         vector              vector associated with best PC.
    '''



    groupOne = X[X_prime[:,0]<0.5]
    groupTwo = X[np.intersect1d(np.where(X_prime[:,0]<2), np.where(X_prime[:,0]>=0.5))]
    groupThree = X[X_prime[:,0]>=2]

    PC = evecs[:, best_PC-1]
    means = np.zeros(3)
    means[0] = np.mean(np.multiply(groupOne,PC))

    means[1] = np.mean(np.multiply(groupTwo,PC))

    means[2] = np.mean(np.multiply(groupThree,PC))

    correct = 0
    for i in range(X.shape[0]):
        vectorised = np.dot(X[i,:], PC)
        vectorised = [vectorised] * 3
        differences = np.absolute(vectorised - means)
        group = np.argmin(differences)

        FRC = X_prime[i,0]
        if group == 0 and FRC < 0.2:
            correct += 1
        elif group == 1 and FRC < 2 and FRC >= 0.5:
            correct += 1
        elif group == 2 and FRC >= 2:
            correct += 1

    score = correct/X.shape[0]
    return score, PC


def calc_score_two_groups(X, evecs, best_PC, X_prime):
    '''Function to calculate the proportion of correctly classified points for a split of 2 groups.
    
    Inputs
        X:          (M x N) array       Normalised data array with variables involved in decomposition
        evecs:      (N x N) array       Array of corresponding eigenvectors (each eigenvector is a column)
        best_PC     int                 the number of the PC best aligned with the FRC
        X_prime:    (M' x N) array      unnormalised data including FRC column.

    Outputs
        score:      float               proportion of correctly classified points.
        PC:         vector              vector associated with best PC.
    '''

    groupOne = X[X_prime[:,0]<0.5]
    groupTwo = X[X_prime[:,0]>=0.5]

    PC = evecs[:, best_PC-1]
    means = np.zeros(2)
    means[0] = np.mean(np.multiply(groupOne,PC))

    means[1] = np.mean(np.multiply(groupTwo,PC))

    correct = 0
    for i in range(X.shape[0]):
        vectorised = np.dot(X[i,:], PC)
        vectorised = [vectorised] * 2
        differences = np.absolute(vectorised - means)
        group = np.argmin(differences)

        FRC = X_prime[i,0]
        if group == 0 and FRC < 0.5:
            correct += 1
        elif group == 1 and FRC >= 0.5:
            correct += 1

    score = correct/X.shape[0]
    return score, PC

def plot_linreg(chosen_sets, x_var, y_var, ax):
    '''Create a scatter plot of datapoints from the chosen datasets for the chosen 2 variables.

    Inputs
        datasets:   dict            all datasets in use
        x_var:      string          parameter along x axis
        y_var:      string          parameter along y axis
    
    Return an (M x 2) array of all the points plotted.
    '''
    datasets =  data.fetch_data(chosen_sets)

    count = 0
    for source in datasets:
        colour = colours[count]
        df = datasets[source][datasets[source][x_var] > -1]
        df = df[df[y_var] > -1]
        x = df[x_var]
        y = df[y_var]
        ax.scatter(x, y, label = source, color = colour)

        slope, intercept, r, p, se = scipy.stats.linregress(x, y)

        ax.plot(x, intercept + slope*x, color = colour, linewidth = 0.5, label='m = {:.2f} \n$R^2 = {:.3f}$'.format(slope, r**2))
        count += 1


    ax.legend()
    ax.set_xlabel('{} {}'.format(x_var, data.units[x_var]))
    ax.set_ylabel('{} {}'.format(y_var, data.units[y_var]))
    ax.set_title('{} vs {}'.format(x_var, y_var))

    # fig.savefig('Graphs/{}/{}.pdf'.format(y_var, 'Cam'), format = 'pdf', transparent=False, bbox_inches='tight')
    plt.show()

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum(SC) over classes
        # SC = sum((x_i - mean_x_c)^2) over i in c.

        # Between class scatter:
        # (weighted sum of distances of class means from overall mean.)
        # SB = sum( n_c * (mean_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        # this is because we want to maximise inter-class directions influence (SB) and minimise
        # influence of intra-class directions (SW)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)