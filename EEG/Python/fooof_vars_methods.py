####################
# Helper functions #
####################
def findAlpha(peaks):
    for peak_params in peaks:
        peak = peak_params[0]
        if 8<peak and peak <13:
            return peak
    return float('nan')

def findAmps(peaks):
    for peak_params in peaks:
        peak = peak_params[0]
        amp  = peak_params[0]
        if 7<peak and peak <13:
            return amp
    return float('nan')
