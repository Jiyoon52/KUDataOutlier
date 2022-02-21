import numpy as np
import sranodec as anom

# Spectral Residual
def SR(args):

    train_data = args.train_data
    signal = train_data.reshape(-1)

    spec = anom.Silency(args.SR_spectral_window_size, args.SR_series_window_size, args.SR_score_window_size)
    score = spec.generate_anomaly_score(signal)
    index_changes = np.where(score > np.percentile(score, args.percentile))[0]

    return index_changes
