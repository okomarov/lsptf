import numpy as np

def binData(*data, IndependentSort=True, NumBins=10, Percentiles=None, Edges=None):
    """Returns indices of the percentile or custom bins to which each value in input array belongs.
    
    Two-dimensional data are sorted by rows either independently, i.e. one by one,
    or conditioning on the bin from the previous sort. In the latter case, order 
    of the `*data` matters, i.e. `data1` is first mapped into bins, then values 
    of `data2` corresponding to `bin=1` (from the `data1`) are mapped into bins,
    etc... 
    Bins are delimited with left inclusion, i.e. lb <= x < ub, except for the 
    last one which also includes the right edge.     
    The lower and upper boundaries of the bins are row-wise percentiles.

    Args:
        data1,...,dataN (numpy.ndarray): array-like datasets to sort by rows. 
            Must have same shape.
             
        IndependentSort (Optional[bool]): alternative is conditional sort. 
            Defaults to True.
               
        NumBins (Optional[int]): number of equal percentiles data is mapped into. 
            Defaults to 10. 
            
        Percentiles (Optional[float]): array of values between [0,100]. It has 
            to be 1-dimensional and monotonic. If supplied, `NumBins` is ignored.  
            Defaults `linspace(0,100, NumBins+1)`.
        
        Edges (Optional[float]): array with absolute bin values. It has to be 
            monotonic. If supplied both `NumBins` and `Percentiles` are ignored. 
            
    Returns:
        out (numpy.ndarray): k by data.shape array of indices where the k-th 
            layer correponds to the k-th `*data`. 
  
    See also: numpy.digitize, numpy.percentile, numpy.linspace
    """


    # Extract edges info
    if Edges is not None:
        opts = {'edges': Edges,
                'nbins': Edges.shape[1]}
    elif Percentiles is not None:
        opts = {'ptiles': Percentiles,
                'nbins' : Percentiles.shape[1]}
    else:
        opts = {'ptiles': np.linspace(0, 100, NumBins+1),
                'nbins' : NumBins}

    # Bin
    if IndependentSort or len(data) == 1:
        bins = _binIndependently(*data, **opts)
    else:
        bins = _binConditionally(None,*data, **opts)

    return bins


def _binIndependently(*data, **opts):
    return np.squeeze([_binCore(x, **opts) for x in data])


def _binConditionally(binmask,*data, **opts):
    """
    Bin within bins, conditioning on previous run. Recursive implementation
    """

    sz = (len(data),) + data[0].shape
    # First run
    if binmask is None:
        bins    = np.zeros(sz, np.int64)
        bins[0] = _binCore(data[0], **opts)
    else:
        # Bin on previously conditioned
        tmp  = np.ma.array(data[0], mask=~binmask, fill_value=np.nan)
        bins = _binCore(tmp.filled(), **opts)

    # Recurse if next level is available
    if sz[0] > 1:
        # Condition on each value of current binning
        for v in np.arange(opts['nbins'])+1:
            binmask          = bins[0] == v
            bins[1, binmask] = _binConditionally(binmask,*data[1:], **opts)[binmask]

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
