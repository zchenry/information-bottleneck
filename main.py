#!/usr/bin/env python
from lib import *
from mi import calc_mi
from utility import pr, tuple2array, MLP

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def train_model(model, train_iter, test_xs, test_ys,
                it, epochs, batchsize, hs_index):
    optimizer = chainer.optimizers.SGD()
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
        pr('ITER {}, EPOCH {}, ACC {:.5f}'.format(it, epoch, acc.data[()]))
    return hiddens_list

def plot_planes(planes, es, epochs):
    for i, epoch in enumerate(es):
        pr('Plotting epoch {:3d}/{}...'.format(epoch, epochs))
        plt.clf()
        for points in planes:
            _xs = np.array([p['ixt'] for p in points])
            _ys = np.array([p['ity'] for p in points])
            _es = np.array([p['epoch'] for p in points])
            mask = _es == i
            plt.scatter(_xs[mask], _ys[mask], color=COLORS[:sum(mask)])
            plt.title('EPOCH {}'.format(epoch))
        plt.savefig('plane_epoch{:03d}.png'.format(epoch))

def run(hs, gpu, iters, epochs, snaps, batchsize):
    planes = []
    train, test = chainer.datasets.get_mnist()
    xp = cp if gpu >=0 else np
    test_xs, test_ys = tuple2array(test, xp)
    size = 100
    test_xs, test_ys = test_xs[:size], test_ys[:size]
    bins = np.linspace(-1, 1, 30)
    Ts_index = np.array(np.linspace(0, 1, snaps) * epochs).astype(np.int)

    for it in range(iters):
        model = MLP(hs)
        if gpu >= 0:
            model.to_gpu()

        train_iter = chainer.iterators.SerialIterator(train,
                                                      batch_size=batchsize)
        Ts_list = train_model(model, train_iter, test_xs, test_ys,
                              it, epochs, batchsize, Ts_index)
        points = calc_mi(test_xs, test_ys[:, None], Ts_list, bins)
        planes.append(points)
    os.system('rm *.png')
    plot_planes(planes, Ts_index[:-1], epochs)
    os.system('convert -delay 30 *.png plane.gif')

    pr('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+',
                        type=int, default=[30, 10, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--snaps', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=128)
    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as cp
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
    run(args.hidden, args.gpu, args.iter, args.epoch,
        args.snaps, args.batchsize)
