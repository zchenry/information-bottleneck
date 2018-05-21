from lib import *

def pr(s, l=100):
    print(' ' * l, end='\r')
    print(s, end='\r')

def tuple2array(data, xp):
    xs = xp.array([d[0] for d in data], dtype=xp.float32)
    ys = xp.array([d[1] for d in data])
    return xs, ys

def mmd(x1, y1, h=0.001):
    g = 1. / (2. * h**2)
    g = 1./ (2. * median(x1)**2)
    kxx = np.mean(rbf_kernel(x1, x1, g))
    kxy = np.mean(rbf_kernel(x1, y1, g))
    g = 1./ (2. * median(y1)**2)
    kyy = np.mean(rbf_kernel(y1, y1, g))
    return kxx - 2. * kxy + kyy

def median(x1):
    sq_dist = pdist(x1)
    pairwise_dists = squareform(sq_dist)**2
    band = np.median(pairwise_dists)
    band = np.sqrt(0.5 * band / np.log(len(x1) + 1))
    return band

def rand_idx(maxi, n=2000):
    return np.array(np.random.rand(2000) * (maxi - 1)).astype(np.int)

def calc_hsics(xs, ys, ts):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_hsic)(ts[e][l], xs, ys, e, l)
        for e in range(len(ts)) for l in range(len(ts[e])))

def calc_hsic(ts, xs, ys, e, t):
    pr('Calculating HSIC for SNAP {} LAYER {}...'.format(e + 1, t + 1))
    m = len(ts)
    kx = rbf_kernel(xs, xs, 1./(2*median(xs)**2))
    ky = rbf_kernel(ys, ys, 1./(2*median(ys)**2))
    l = rbf_kernel(ts, ts, 1./(2*median(ts)**2))
    h = np.ones((m, m)) * (1. - 1./m)
    hlh = h @ l @ h
    params = {}
    params['ixt'] = np.trace(kx @ hlh) / m**2
    params['ity'] = np.trace(ky @ hlh) / m**2
    params['epoch'] = e
    params['layer'] = t
    return params

def calc_kl(xs, ys, ts, e, l):
    pr('Calculating KL for SNAP {} LAYER {}...'.format(e + 1, l + 1))
    _, pxs = np.unique(xs, return_counts=True)
    log_px = np.sum(np.log(pxs) - np.log(np.sum(pxs)))
    log_px *= len(ts)
    _, pys = np.unique(ys, return_counts=True)
    log_py = np.sum(np.log(pys) - np.log(np.sum(pys)))
    log_py *= len(ts)

    bts = np.ascontiguousarray(ts).view(np.dtype((np.void, ts.dtype.itemsize * ts.shape[1])))
    unique_array, unique_indices, unique_inverse_t, unique_counts = \
        np.unique(bts, return_index=True, return_inverse=True, return_counts=True)

    px_t, py_t = [], []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_t == i
        px_t_current = np.mean(xs[indexs, :], axis=0)
        px_t.append(px_t_current)
        py_t_current = np.mean(ys[indexs, :], axis=0)
        py_t.append(py_t_current)
    px_t = np.array(px_t).T
    py_t = np.array(py_t).T

    params = {}
    params['ixt'] = log_px - np.sum(np.log(px_t))
    params['ity'] = log_py - np.sum(np.log(py_t))
    params['epoch'] = e
    params['layer'] = l
    return params

def calc_mmd(xs, ys, ts, e, l):
    pr('Calculating MMD for SNAP {} LAYER {}...'.format(e + 1, l + 1))

    xt = xs[rand_idx(xs.shape[0])]
    yt = ys[rand_idx(ys.shape[0])]

    bts = np.ascontiguousarray(ts).view(np.dtype((np.void, ts.dtype.itemsize * ts.shape[1])))
    unique_array, unique_indices, unique_inverse_t, unique_counts = \
        np.unique(bts, return_index=True, return_inverse=True, return_counts=True)

    px_t, py_t = [], []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_t == i
        px_t_current = np.mean(xs[indexs, :], axis=0)
        px_t.append(px_t_current)
        py_t_current = np.mean(ys[indexs, :], axis=0)
        py_t.append(py_t_current)
    # px_t = np.array(px_t).T
    # py_t = np.array(py_t).T

    sxt = np.random.binomial(1, np.array(px_t)[rand_idx(xs.shape[0])])
    syt = np.array(py_t)[rand_idx(ys.shape[0])]

    '''
    yt_idx = rand_idx(ys.shape[0])
    yt_left, yt_right = ys[yt_idx], ts[yt_idx]
    yt = np.concatenate((yt_left, yt_right), axis=1)

    sxt_left = xs[rand_idx(xs.shape[0])]
    sxt_right = ts[rand_idx(ts.shape[0])]
    sxt = np.concatenate((sxt_left, sxt_right), axis=1)

    syt_left = ys[rand_idx(ys.shape[0])]
    syt_right = ts[rand_idx(ts.shape[0])]
    syt = np.concatenate((syt_left, syt_right), axis=1)
    '''
    params = {}
    params['ixt'] = mmd(xt, sxt)
    params['ity'] = mmd(yt, syt)
    params['epoch'] = e
    params['layer'] = l
    return params

def calc_mmds(xs, ys, tss):
    pr('Calculating MMD ...')
    # calc_mmd(xs, ys, tss[0][0], 0, 0)
    return [Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_mmd)(xs, ys, tss[e][l], e, l)
        for e in range(len(tss)) for l in range(len(tss[e])))]

def calc_kls(xs, ys, tss):
    pr('Calculating KL ...')
    return [Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_kl)(xs, ys, tss[e][l], e, l)
        for e in range(len(tss)) for l in range(len(tss[e])))]

class MLP(chainer.Chain):
    def __init__(self, hs, out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.hs = hs + [10]
            for i, h in enumerate(self.hs):
                n_in = 784 if i == 0 else hs[i-1]
                setattr(self, 'w{}'.format(i),
                        chainer.Parameter(
                            initializers.Normal(scale=np.sqrt(1./n_in)), (h, n_in)))
                setattr(self, 'b{}'.format(i),
                        chainer.Parameter(initializers.Zero(), (h,)))

    def forward(self, x, keep_val=False):
        if keep_val:
            vals = []

        for i in range(len(self.hs) - 1):
            w = getattr(self, 'w{}'.format(i))
            b = getattr(self, 'b{}'.format(i))
            x = F.arctan(F.linear(x, w, b))
            # x = F.relu(F.linear(x, w, b))
            if keep_val:
                vals.append(x)

        w = getattr(self, 'w{}'.format(len(self.hs) - 1))
        b = getattr(self, 'b{}'.format(len(self.hs) - 1))
        output = F.linear(x, w, b)
        if keep_val:
            vals.append(output)

        if keep_val:
            return output, vals
        else:
            return output

    def loss(self, x, y):
        return F.softmax_cross_entropy(self.forward(x), y)

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 3)
            self.conv2 = L.Convolution2D(16, 32, 3)
            self.conv3 = L.Convolution2D(32, 64, 3)
            self.l1 = L.Linear(256, 30)
            self.l2 = L.Linear(30, 20)
            self.l3 = L.Linear(20, 15)
            self.l4 = L.Linear(15, 12)
            self.l5 = L.Linear(12, 10)

    def forward(self, x, keep_val=False):
        x = F.reshape(x, (-1, 1, 28, 28))
        #h1 = F.relu(self.conv1(x))
        h1 = F.arctan(self.conv1(x))
        h2 = F.max_pooling_2d(h1, 2)
        #h2 = F.relu(self.conv2(h2))
        h2 = F.arctan(self.conv2(h2))
        h3 = F.max_pooling_2d(h2, 2)
        #h3 = F.relu(self.conv3(h3))
        h3 = F.arctan(self.conv3(h3))
        h4 = F.max_pooling_2d(h3, 2)
        #h5 = F.relu(self.l1(h4))
        #h6 = F.relu(self.l2(h5))
        #h7 = F.relu(self.l3(h6))
        #h8 = F.relu(self.l4(h7))
        h5 = F.arctan(self.l1(h4))
        h6 = F.arctan(self.l2(h5))
        h7 = F.arctan(self.l3(h6))
        h8 = F.arctan(self.l4(h7))
        y = self.l5(h8)

        if keep_val:
            return y, [F.reshape(h4, (h4.shape[0], -1)), h5, h6, h7, h8]
        else:
            return y

    def loss(self, x, y):
        return F.softmax_cross_entropy(self.forward(x), y)
