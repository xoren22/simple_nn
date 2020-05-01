import numpy as np
import matplotlib.pyplot as plt

def to_categorical(y):
    uniques = np.unique(y)
    sparse_cat = np.zeros((len(y), np.max(y)+1))
    sparse_cat[np.arange(len(y)), y] = True
    res = sparse_cat[:, uniques]
    
    return res

def dist(x, y, title=None):
	x = x.reshape(-1)
	plt.hist(x[y[:,0]], bins=50, alpha=.5)
	plt.hist(x[y[:,1]], bins=50, alpha=.5)
	if title:
		plt.title(title)
	plt.show()

def bug_print(string):
	print("*"*40)
	print("*"*40)
	print("\n")
	print(string)
	print("\n")
	print("*"*40)
	print("*"*40)


def plot_acts(acts, y): 
	"""
	fig, axarr = plt.subplots(nrows=4, ncols=neurons, sharex=True, sharey=True)
	fig.set_size_inches(15, 10)

	fig.text(0.5, -0.03, 'Neuron', fontsize=20, ha='center')
	fig.text(-0.03, 0.5, 'Layer',fontsize=20, va='center', rotation='vertical')


	for i in range(layers*neurons):        
	    axarr[i//neurons, i%neurons].set_xlim(-1.1, 1.1)
	    axarr[i//neurons, i%neurons].set_ylim(-5, 5)     
	    axarr[i//neurons, i%neurons].scatter(x=layer_outs[i//neurons][:,i%neurons],y=rand[:500],s=54,
	                                         c=y_train[inds][:500],alpha=0.5)

	plt.tight_layout() 
	plt.show()	




for epoch in range(epochs): 
    layer_outs = [get_layer_outputs(x[:500], layer=lyr) for lyr in range(layers)]
    clear_output(wait=True)
    fig, axarr = plt.subplots(nrows=4, ncols=neurons, sharex=True, sharey=True)
    fig.set_size_inches(15, 10)
    
    fig.text(0.5, -0.03, 'Neuron', fontsize=20, ha='center')
    fig.text(-0.03, 0.5, 'Layer',fontsize=20, va='center', rotation='vertical')
   
    
    for i in range(layers*neurons):        
        axarr[i//neurons, i%neurons].set_xlim(-1.1, 1.1)
        axarr[i//neurons, i%neurons].set_ylim(-5, 5)     
        axarr[i//neurons, i%neurons].scatter(x=layer_outs[i//neurons][:,i%neurons],y=rand[:500],s=54,
                                             c=y_train[inds][:500],alpha=0.5)

    plt.tight_layout() 
    plt.show()

    model.fit(x, y, epochs=1, batch_size=50, verbose=1)
    """
	pass