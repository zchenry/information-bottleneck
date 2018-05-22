#!/usr/bin/env python
from lib import *
from mi import calc_mi
from utility import pr, tuple2array, MLP, calc_mmds

def train_model(model, train_iter, test_xs, test_ys,
                epochs, batchsize, hs_index, lr, hs, thre):
    optimizer = chainer.optimizers.SGD(lr)
    optimizer.setup(model)

    for i, h in enumerate(hs):
       if h > thre:
            getattr(model, 'w{}'.format(i)).update_rule.enabled = False
            getattr(model, 'b{}'.format(i)).update_rule.enabled = False

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
            print('EPOCH {}, ACC {:.5f}'.format(epoch + 1, acc.data[()]))
    return hiddens_list

def plot_planes(points, es, hs, feature):
    _xs = np.array([p['ixt'] for p in points])
    _ys = np.array([p['ity'] for p in points])
    _ls = np.array([p['layer'] for p in points])

    plt.clf()
    for l in np.sort(np.unique(_ls)):
        mask = _ls == l
        plt.plot(es, _xs[mask], label=str(l))
        plt.legend()
    plt.savefig('{}_x.png'.format(feature))

    plt.clf()
    for l in np.sort(np.unique(_ls)):
        mask = _ls == l
        plt.plot(es, _ys[mask], label=str(l))
        plt.legend()
    plt.savefig('{}_y.png'.format(feature))

def run(hs, gpu, epochs, snaps, batchsize, lr, thre):
    planes = []
    train, test = chainer.datasets.get_mnist()
    xp = cp if gpu >=0 else np
    test_xs, test_ys = tuple2array(test, xp)
    size = 10000
    test_xs = test_xs[:size]; test_ys = test_ys[:size]
    bins = np.linspace(-1, 1, 30)
    Ts_index = np.array(np.linspace(0, 1, snaps) * (epochs - 1)).astype(np.int)
    least = 10
    if Ts_index[1] > least:
        Ts_index = np.unique(np.append(np.array(range(least)), Ts_index[1:]))

    model = MLP(hs)
    if gpu >= 0:
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(train,
                                                  batch_size=batchsize)
    Ts_list = train_model(model, train_iter, test_xs, test_ys,
                          epochs, batchsize, Ts_index, lr, hs, thre)
    feature = '{}_thre{}_epoch{}_bs{}_lr{}'.format(
        '_'.join([str(h) for h in hs]), thre, epochs, batchsize, lr)
    points = calc_mi(to_cpu(test_xs), to_cpu(test_ys)[:, None],
                     Ts_list, bins)
    plot_planes(points, Ts_index, hs, feature)
    pr('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+',
                        type=int, default=[30, 20, 15, 14, 13, 12, 11])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--snaps', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--thre', type=int, default=128)
    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as cp
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
    run(args.hidden, args.gpu, args.epoch,
        args.snaps, args.batchsize, args.lr, args.thre)
