# from https://github.com/ravidziv/IDNNs

from lib import *
from utility import pr

def KL(a, b):
    return np.nansum(np.multiply(a, np.log(np.divide(a, b+np.spacing(1)))), axis=1)

def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
    """Calculate the p(x|T) and p(y|T)"""
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False,
                  return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    sum_counts = float(sum(unique_counts_internal))
    p_x_ts[indexes_x] = unique_counts_internal / sum_counts
    return p_x_ts, p_y_ts

def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False,
                  return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X

def calc_condtion_entropy(px, t_data, unique_inverse_x):
    H2X_array = np.array(
        Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
                for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X

def calc_information_from_mat(px, py, ps2, data, unique_inverse_x,
                              unique_inverse_y, unique_array):
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IX = H2 - H2X
    IY = H2 - H2Y
    return IX, IY

def calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                              len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y):
    bins = bins.astype(np.float32)
    nbins = bins.shape[0]
    digitized = bins[np.digitize(
        np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False,
                  return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    if False:
        pxy_given_T = np.array(
            [calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
        )
        p_XgT = np.vstack(pxy_given_T[:, 0])
        p_YgT = pxy_given_T[:, 1]
        p_YgT = np.vstack(p_YgT).T
        DKL_YgX_YgT = np.sum(
            [KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
        H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
    IXT, ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized,
                                         unique_inverse_x, unique_inverse_y,
                                         unique_array)
    return IXT, ITY

def calc_information_for_layer_with_other(data, bins, unique_inverse_x,
                                          unique_inverse_y, label, b, b1,
                                          len_unique_a, pxs, p_YgX, pys1, e, l,
                                          percent_of_sampling=50):
    pr('Calculating MI for SNAP {} LAYER {}...'.format(e + 1, l + 1))
    IXT, ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                                         len_unique_a, p_YgX, unique_inverse_x,
                                         unique_inverse_y)
    number_of_indexs = int(data.shape[1] * (1. / 100 * percent_of_sampling))
    indexs_of_sampls = np.random.choice(data.shape[1], number_of_indexs,
                                        replace=False)
    if percent_of_sampling != 100:
        sampled_data = data[:, indexs_of_sampls]
        sampled_IXT, sampled_ITY = calc_information_sampling(
            sampled_data, bins, pys1, pxs, label, b, b1,
            len_unique_a, p_YgX, unique_inverse_x, unique_inverse_y)

    params = {}
    params['ixt'] = IXT
    params['ity'] = ITY
    params['epoch'] = e
    params['layer'] = l
    return params

def extract_probs(label, x):
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1])))

    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]

    b1 = np.ascontiguousarray(unique_a).view(
        np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))

    pxs = unique_counts / float(np.sum(unique_counts))

    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T

    b_y = np.ascontiguousarray(label).view(
        np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True,
                  return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs

def calc_mi(xs, ys, ts, bins):
    ys = ys.astype(np.float32)
    pys, pys1, py_x, b1, b, \
        unique_a, unique_x, unique_y, pxs = extract_probs(ys, xs)
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_information_for_layer_with_other)(
            ts[e][l], bins, unique_x, unique_y, ys,
            b, b1, len(unique_a), pxs, py_x, pys1, e, l)
        for e in range(len(ts)) for l in range(len(ts[e])))
