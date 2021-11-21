import argparse
from utils import prepare_save_dir, create_logger, save_logger
from STELLAR import STELLAR

def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--dataset', default='codex', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='STELLAR')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--input-dim', type=int, default=40)
    parser.add_argument('--num-heads', type=int, default=20)
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    args.name = '_'.join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    stellar = STELLAR(args, labeled_X, labeled_y, unlabeled_X, labeled_pos, unlabeled_pos)
    stellar.train()
    _, results = stellar.pred()


if __name__ == '__main__':
    main()
