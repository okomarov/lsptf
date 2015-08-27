import numpy as np

def binData(*data, IndependentSort=True, NumBins=10, Percentiles=None, Edges=None):

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
