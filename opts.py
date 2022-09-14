import argparse

def opts():
    parser = argparse.ArgumentParser(description='SRDC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #model
    # datasets
    parser.add_argument('--data_path_source', type=str, default='./data/datasets/Office31/', help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default='./data/datasets/Office31/', help='root of target training set')
    parser.add_argument('--data_path_target_t', type=str, default='./data/datasets/Office31/', help='root of target test set')
    parser.add_argument('--src', type=str, default='amazon', help='source training set')
    parser.add_argument('--tar', type=str, default='webcam_half', help='target training set')
    parser.add_argument('--tar_t', type=str, default='webcam_half2', help='target test set')
    parser.add_argument('--num_classes', type=int, default=31, help='class number')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')

    # specific optimization options
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of K-means')
    parser.add_argument('--cluster_kernel', type=str, default='rbf', help='kernel to choose when using kernel K-means')
    parser.add_argument('--gamma', type=float, default=None, help='bandwidth for rbf or polynomial kernel when using kernel K-means')
    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoints path to resume')
    parser.add_argument('--log', type=str, default='./checkpoints/office31', help='log folder')
    parser.add_argument('--stop_epoch', type=int, default=200, metavar='N', help='stop epoch for early stop (default: 200)')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained model')
    # i/o
    parser.add_argument('--print_freq', type=int, default=1, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained_path', type=str, default="", help='path of pretrained model')

    args = parser.parse_args()
    args.log = args.log + '_adapt_' + args.src + '2' + args.tar + '_bs' + str(args.batch_size) + '_' + args.arch + '_lr' + str(args.lr)

    return args
