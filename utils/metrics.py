import numpy as np

def abs_rel(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    return np.mean(np.abs(target[mask] - prediction[mask]) / target[mask])

def mae(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    return np.mean(np.abs(prediction[mask] - target[mask]))

def sq_rel(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    sq_rel = np.mean(((target[mask] - prediction[mask]) ** 2) / target[mask])
    return sq_rel

def si_log(target, prediction):
    # https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf Section 3.2
    mask = np.isfinite(target) & (target > 0)
    num_vals = mask.sum()
    log_diff = np.log(prediction[mask]) - np.log(target[mask])
    si_log_unscaled = np.sum(log_diff**2) / num_vals - (np.sum(log_diff)**2) / (num_vals**2)
    si_log_score = np.sqrt(si_log_unscaled)*100
    return si_log_score

def calculate_metrics(depth_annotation, depth_prediction):
        mae_score = mae(depth_annotation, depth_prediction)
        abs_rel_scre = abs_rel(depth_annotation, depth_prediction)
        sq_rel_score = sq_rel(depth_annotation, depth_prediction)
        si_log_score = si_log(depth_annotation, depth_prediction)
        metrics = {
                    "si_log": float(si_log_score),
                    "abs_rel": float(abs_rel_scre),
                    "sq_rel": float(sq_rel_score),
                    "mae": float(mae_score)
                  }
        return  metrics

