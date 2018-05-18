#!/usr/bin/env python
from lib import *
from mi import calc_mi
from utility import pr, tuple2array, MLP

COLORS = ['y', 'r', 'k', 'g', 'b', 'c', 'm', 'grey']

def train_model(model, train_iter, test_xs, test_ys,
                it, epochs, batchsize, hs_index, lr):
    optimizer = chainer.optimizers.SGD(lr)
    optimizer.setup(model)

    hiddens_list = []
    for epoch in range(epochs):
        train_xs, train_ys = tuple2array(train_iter.next(), model.xp)
        loss = model.loss(train_xs, train_ys)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        pred_ys, hiddens_gpu = model.forward(test_xs, keep_val=True)
        acc = F.accuracy(pred_ys, test_ys)
        acc.to_cpu()
        if epoch in hs_index:
            hiddens = []
            for hidden in hiddens_gpu:
                hidden.to_cpu()
                hiddens.append(hidden.data[()])
            hiddens_list.append(hiddens)
        if (epoch + 1) % 100 == 0:
            print('EPOCH {}, ACC {:.5f}'.format(epoch + 1, acc.data[()]))
    return hiddens_list

def plot_planes(planes, es, hs, feature):
    '''
    for i, epoch in enumerate(es):
        plt.clf()
        for points in planes:
            _xs = np.array([p['ixt'] for p in points])
            _ys = np.array([p['ity'] for p in points])
            _es = np.array([p['epoch'] for p in points])
            mask = _es == i
            plt.scatter(_xs[mask], _ys[mask], color=COLORS[:sum(mask)])
            plt.title('EPOCH {}'.format(epoch))
            plt.xlim(12.5, 13.5)
            plt.ylim(3.15, 3.35)
        plt.savefig('plane_epoch{:07d}.png'.format(epoch))

    plt.clf()
    for i, epoch in enumerate(es):
        for points in planes:
            _xs = np.array([p['ixt'] for p in points])
            _ys = np.array([p['ity'] for p in points])
            _es = np.array([p['epoch'] for p in points])
            mask = _es == i
            plt.scatter(_xs[mask], _ys[mask], color=COLORS[:sum(mask)])
    plt.xlim(12, 14)
    plt.ylim(3, 3.35)
    plt.title('ALL')
    plt.savefig('plane.png')
    '''
    for it, points in enumerate(planes):
        plt.clf()
        _xs = np.array([p['ixt'] for p in points])
        _ys = np.array([p['ity'] for p in points])
        _ls = np.array([p['layer'] for p in points])
        for l in range(len(hs)):
            mask = _ls == l
            plt.plot(_xs[mask], _ys[mask], color=COLORS[l])
        plt.savefig('{}.png'.format(feature))

def run(hs, gpu, iters, epochs, snaps, batchsize, lr):
    planes = []
    train, test = chainer.datasets.get_mnist()
    xp = cp if gpu >=0 else np
    test_xs, test_ys = tuple2array(test, xp)
    size = 10000
    test_xs = test_xs[:size]; test_ys = test_ys[:size]
    bins = np.linspace(-1, 1, 30)
    Ts_index = np.array(np.linspace(0, 1, snaps) * epochs).astype(np.int)

    for it in range(iters):
        model = MLP(hs)
        if gpu >= 0:
            model.to_gpu()

        train_iter = chainer.iterators.SerialIterator(train,
                                                      batch_size=batchsize)
        Ts_list = train_model(model, train_iter, test_xs, test_ys,
                              it, epochs, batchsize, Ts_index, lr)
        points = calc_mi(to_cpu(test_xs), to_cpu(test_ys)[:, None],
                         Ts_list, bins)
        planes.append(points)

    feature = '{}_epoch{}_bs{}_lr{}'.format(
        '_'.join([str(h) for h in hs]), epochs, batchsize, lr)
    np.save('{}.npy'.format(feature), [planes, Ts_index[:-1]], hs)
    plot_planes(planes, Ts_index[:-1], hs, feature)
    pr('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+',
                        type=int, default=[30, 20, 15, 14, 13, 12, 11])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=4000)
    parser.add_argument('--snaps', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as cp
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
    run(args.hidden, args.gpu, args.iter, args.epoch,
        args.snaps, args.batchsize, args.lr)
