import argparse
from utils import get_loader
from train import train

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Script')
    
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum for training')
    parser.add_argument('-n', '--num_epochs', type=int, default=300)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    train_loader, test_loader, train_fp, test_fp, test_values = get_loader('train.pkl', 'test.pkl', args)

    train(args, train_loader, test_loader, train_fp, test_fp, test_values)

    

