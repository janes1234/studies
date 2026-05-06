#
#
#
#

import torch
from numpy import load
import matplotlib.pyplot as plt

####################################################################
#            (1.a) Import the data and print out data              #
#                           dimensionality                         #
####################################################################

# import the data
def read_npz(filename):
    """
    Reads the MNIST dataset from a numpy archive.
    
    Arguments:
    ----------
        filename: str
            - Filename for numpy archive

    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        Y : np.ndarray
            - One-hot encoded target class vectors

    Notes:
    ------
        .npz files are compressed binary formats created using:
            - np.savez_compressed("filename.npz",arr1=arr1,arr2=arr2,...)
        Which can be loaded by:
            - data = np.load(filename)
        Returing a dict with keys -> arr1,arr2,....
            - arr1 = data['arr1']
            - Values in dictionary are np.ndarray
    """

    data = load(filename)

    X = torch.tensor(data['X'],dtype=torch.float32)
    Y = torch.tensor(data['Y'])

    Y = torch.argmax(Y,-1).type(torch.long)

    return X,Y

####################################################################
# (3) For calculating the number of correct predictions in a batch #
####################################################################

def hits(y_hat,y):
    """Calculate the number of correct predictions in a batch

    Arguments:
    ----------
        y_hat : torch.tensor
            - Model scores matrix of dimensionality (N,K)
                where N : number of examples in batch
                      K : number of possible classes
        y : torch.tensor
            - Ground truth vector of dimensionality (N)
              where each value is the correct class index

    Returns:
    --------
        hits : torch.tensor
            - Number of correct predictions, i.e. number of times
            the index corresponding to the largest score in y_hat
            equals the ground truth y
    """
    # get the predicted ids for each example
    pred_ids = torch.argmax(y_hat,-1)

    # check where the predictions match the targets
    corr = torch.eq(pred_ids,y)

    # return the number of times the prediction is correct
    hits = torch.sum(corr)

    return hits

####################################################################
# (4) Function to plot the 
####################################################################

def plot_filters(X,H,z_1,v_1):
    """Function to visualise the filters, correlations and
        output matrices for the first layer a CNN model with C filters.

    Saves plot in filters.png

    Arguments:
    ----------
        X : torch.tensor                                   
            - Feature vector with D dimensions             
        H : torch.tensor                                   
            - CNN weights/filters with dimensions          
                (C,M_3,M_1,M_2)                              
        z_1 : torch.tensor                                 
            - CNN layer one output after bias with         
                dimensions (B,C,Q_1,Q_2)                     
        v_1 : torch.tensor                                 
            - CNN layer one output after activation        
                function with dimensions (B,C,Q_1,Q_2)

        D : total number of input features
        C : number of filters/channels in CNN
        M : kernel sizes
        B : number of examples in minibatch
    """

    C = H.shape[0] # number of CNN filters

    if C > 15:
        print("*" * 40)
        print("suggest manually adjusting which filters you are about to view,")
        print("because the filters are horizontally stacked, stacking more than 15 could make it quite small")
        print("*" * 40)
    
    fig, axs = plt.subplots(4,C+1,figsize=(C+1,4))

    for i in range(C):
        axs[0,i+1].imshow(X.reshape(28,28))

    for i in range(C):
        axs[1,i+1].imshow(H[i,0,:,:].detach().numpy())

    for i in range(C):
        axs[2,i+1].imshow(z_1[0,i,:,:].detach().numpy())

    for i in range(C):
        axs[3,i+1].imshow(v_1[0,i,:,:].detach().numpy())

    plt.text(0.5, 0.5, "Input \nImage", horizontalalignment='center',
         verticalalignment='center', transform=axs[0,0].transAxes)
    plt.text(0.5, 0.5, "Filter", horizontalalignment='center',
            verticalalignment='center', transform=axs[1,0].transAxes)
    plt.text(0.5, 0.5, "$Z[1]$", horizontalalignment='center',
            verticalalignment='center', transform=axs[2,0].transAxes)
    plt.text(0.5, 0.5, "$V[1]$", horizontalalignment='center',
            verticalalignment='center', transform=axs[3,0].transAxes)

    for ax in fig.axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("filters.png", bbox_inches='tight')