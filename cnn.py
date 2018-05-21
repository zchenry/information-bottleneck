#!/usr/bin/env python
from lib import *
from mi import calc_mi
from utility import pr, tuple2array, CNN, calc_mmds, calc_kls

COLORS = ['y', 'r', 'k', 'g', 'b', 'c', 'm', 'grey']

def train_model(model, train_iter, test_xs, test_ys,
                epochs, batchsize, hs_index, lr):
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

def plot_planes(points, es, feature):
    print('Plotting')
    if len(points) == 1: points = points[0]
    plt.clf()
    _xs = np.array([p['ixt'] for p in points])
    _ys = np.array([p['ity'] for p in points])
    _ls = np.array([p['layer'] for p in points])
    for l in range(len(np.unique(_ls))):
        mask = _ls == l
        plt.plot(_xs[mask], _ys[mask], color=COLORS[l], label=str(l+1))
        plt.plot(_xs[mask][0], _ys[mask][0], color=COLORS[l], marker='x')
    plt.legend()
    plt.savefig('{}.png'.format(feature))

def run(gpu, epochs, snaps, batchsize, lr):
    train, test = chainer.datasets.get_mnist()
    xp = cp if gpu >=0 else np
    test_xs, test_ys = tuple2array(test, xp)
    size = 10000
    test_xs = test_xs[:size]; test_ys = test_ys[:size]

    bins = np.linspace(-1, 1, 30)
    Ts_index = np.array(np.linspace(0, 1, snaps) * epochs).astype(np.int)

    model = CNN()
    if gpu >= 0:
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(train,
                                                  batch_size=batchsize)
    Ts_list = train_model(model, train_iter, test_xs, test_ys,
                          epochs, batchsize, Ts_index, lr)
    points = calc_mi(to_cpu(test_xs), to_cpu(test_ys)[:, None], Ts_list, bins)
    # points = calc_mmds(to_cpu(test_xs), to_cpu(test_ys)[:, None], Ts_list)
    feature = 'cnn_epoch{}_bs{}_lr{}'.format(epochs, batchsize, lr)
    plot_planes(points, Ts_index[:-1], feature)
    pr('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--snaps', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as cp
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
    run(args.gpu, args.epoch, args.snaps, args.batchsize, args.lr)
