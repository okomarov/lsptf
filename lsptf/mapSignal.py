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

def _binCore(data, ptiles=None, edges=None, nbins=None):

    nanmask = np.isnan(data)

    if edges is None:
        # Ensure monotonically increasing percentiles, i.e. convert data to 
        # integer ranks. Applied by rows
        data          = data.argsort().argsort()
        data          = data.astype(float)
        data[nanmask] = np.nan

    # Return bin to which data belongs
    bins = np.zeros(data.shape, np.int64)
    for r in np.arange(data.shape[0])[np.any(~nanmask,axis=1)]:
        notnan = ~nanmask[r, :]
        if edges is None:
            # TODO: add matlab compatible percentile function? Note that matlab uses midpoint interpolation for all but the extreme percentiles!
            edge = np.percentile(data[r, notnan], ptiles)
        else:
            edge = edges[r, :]

        bins[r, notnan] = np.digitize(data[r, notnan], edge)

    # Make sure last bin is lb <= x <= ub
    bins[bins == nbins+1] = nbins

    return bins
