import numpy as np

def binData(*args, IndipendentSort=True, NumBins=10, Percentiles=None, Edges=None):
    
    if Edges is not None:
        opts = {'edges':Edges}
    elif Percentiles is not None:
        opts = {'ptiles':Percentiles}
    else:
        opts = {'ptiles':np.linspace(0,100, NumBins+1)}
    
    
    if IndipendentSort:
        bins = _binIndipendently(*args, **opts)
    else:
        bins = _binConditionally(*args, **opts)

    return bins

def _binIndipendently(*data, **opts):
    return [_binCore(x, **opts) for x in data]

def _binConditionally(*data, **opts):
    pass

def _binCore(data, ptiles=None, edges=None):
    
    if edges is None:
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