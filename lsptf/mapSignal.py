import numpy as np

def _mapCore(data,**kwargs):
    
    # Use supplied edges
    if kwargs.get('HasEdges',False):
        edges = kwargs.get('Edges')
    
    # Partition in percentiles
    else:
        nbins  = kwargs.get('NumBins')
        ptiles = np.linspace(0,100,nbins)
        
        # Ensure monotonically increasing percentiles, i.e. convert data to 
        # integer ranks. Applied by rows
        nanmask       = np.isnan(data)
        data          = data.argsort().argsort()
        data          = data.astype(float)
        data[nanmask] = np.nan
        edges         = np.nanpercentile(data,ptiles,axis=1)
    
    # Return bin to which data belongs
    bins = np.zeros(data.shape,np.int64)
    #bins = np.array([np.digitize(data[r,:], edges[r,:]) for r in np.arange(data.shape[0]) if any(~nanmask[r,:])]) 
    for r in np.arange(data.shape[0]):
        notnan = ~nanmask[r,:]
        if any(notnan):
            bins[r,notnan] = np.digitize(data[r,notnan], edges[r,:])
    
    return bins