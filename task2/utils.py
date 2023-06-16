import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MLP

def unpack(path_train, path_test, args):
    data_input = open(path_train,'rb')
    read_data = pickle.load(data_input)
    train_data = np.array(read_data)
    data_input.close()
    train_data = train_data.item()
    train_fp = np.unpackbits(train_data['packed_fp'], axis=1)
    train_fp = torch.asarray(train_fp)
    train_values = train_data['values']
    
    data_input = open(path_test,'rb')
    read_data = pickle.load(data_input)
    test_data = np.array(read_data)
    data_input.close()
    test_data = test_data.item()
    test_fp = np.unpackbits(test_data['packed_fp'], axis=1)
    test_fp = torch.asarray(test_fp)
    test_values = test_data['values']
    print(max(test_values), min(test_values))
    
    train_dataset = TensorDataset(train_fp, train_values)
    test_dataset = TensorDataset(test_fp, test_values)

    train_loader = DataLoader(train_dataset, args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch, shuffle=True)

def trial():
    a = torch.asarray([1, 2, 3, 4])
    b = torch.asarray([2, 4, 6, 16])
    print(torch.sum(torch.abs(a - b) / b))


def sample(path_train, path_test):
    data_input = open(path_train,'rb')
    read_data = pickle.load(data_input)
    train_data = np.array(read_data)
    data_input.close()
    train_data = train_data.item()
    train_fp = np.unpackbits(train_data['packed_fp'], axis=1)
    train_values = train_data['values']

    data_input = open(path_test,'rb')
    read_data = pickle.load(data_input)
    test_data = np.array(read_data)
    data_input.close()
    test_data = test_data.item()
    test_fp = np.unpackbits(test_data['packed_fp'], axis=1)
    test_values = test_data['values']
    
    # sample_loc = np.random.randint()
    

    N = 3000
    sample_size = 10
    min_value = 0
    max_value = 398580

    samples = []
    for _ in range(N):
        sample = np.random.choice(np.arange(min_value, max_value), size=sample_size, replace=False)
        samples.append(sample)

    samples_array = np.array(samples)
    train_dataset = train_fp[samples_array.flatten()].reshape(N, 10, -1)
    train_value = train_values[samples_array.flatten()].reshape(N).numpy()
    train_value = np.sum(train_value, axis=1)

    N = 1000
    sample_size = 10
    min_value = 0
    max_value = 125239
    
    samples = []
    for _ in range(N):
        sample = np.random.choice(np.arange(min_value, max_value), size=sample_size, replace=False)
        samples.append(sample)

    samples_array = np.array(samples)
    test_dataset = test_fp[samples_array.flatten()].reshape(N, 10, -1)
    test_value = test_values[samples_array.flatten()].reshape(N).numpy()
    test_value = np.sum(test_value, axis=1)

    print(train_dataset.shape, train_value.shape)
    print(test_dataset.shape, test_value.shape)
    np.save('train', train_dataset)
    np.save('test', test_dataset)
    np.save('train_value', train_value)
    np.save('test_value', test_value)


def get_multi_mole(args):
    train_data = torch.asarray(np.load('train.npy').reshape(30000, -1)).float()
    train_value = torch.asarray(np.load('train_value.npy').reshape(30000, -1))
    test_data = torch.asarray(np.load('test.npy').reshape(10000, -1)).float()
    test_value = torch.asarray(np.load('test_value.npy').reshape(10000, -1))

    train_dataset = TensorDataset(train_data, train_value)
    test_dataset = TensorDataset(test_data, test_value)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)

    return train_loader, test_loader

def get_loaded():
    model = MLP()
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()
    model = model.cuda()
    # data_input = open('train.pkl','rb')
    # read_data = pickle.load(data_input)
    # train_data = np.array(read_data)
    # data_input.close()
    # train_data = train_data.item()
    # train_fp = np.unpackbits(train_data['packed_fp'], axis=1)
    # train_values = train_data['values']

    data_input = open('test.pkl','rb')
    read_data = pickle.load(data_input)
    test_data = np.array(read_data)
    data_input.close()
    test_data = test_data.item()
    test_fp = np.unpackbits(test_data['packed_fp'], axis=1)
    test_values = test_data['values']
    
    # sample_loc = np.random.randint()
    

    # N = 30000
    # sample_size = 10
    # min_value = 0
    # max_value = 398580

    # samples = []
    # for _ in range(N):
    #     sample = np.random.choice(np.arange(min_value, max_value), size=sample_size, replace=False)
    #     samples.append(sample)

    # samples_array = np.array(samples)
    # train_dataset = train_fp[samples_array.flatten()].reshape(N, 10, -1)
    # train_value = train_values[samples_array.flatten()].reshape(N, -1).numpy()
    # train_value = np.sum(train_value, axis=1)

    # train_dataset = train_fp[samples_array.flatten()]
    # train_output = train

    N = 10000
    sample_size = 10
    min_value = 0
    max_value = 125239
    
    samples = []
    for _ in range(N):
        sample = np.random.choice(np.arange(min_value, max_value), size=sample_size, replace=False)
        samples.append(sample)

    samples_array = np.array(samples)
    test_dataset = test_fp[samples_array.flatten()]
    test_dataset = torch.asarray(test_dataset).float().cuda()

    test_value = test_values[samples_array.flatten()]
    test_value = torch.asarray(test_value).view(10000, 10)
    test_value = torch.sum(test_value, dim=1)
    test_value = torch.asarray(test_value).cuda()
    test_output = model(test_dataset).view(10000, 10)
    test_output = torch.sum(test_output, dim=1).cuda()
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(test_value, test_output)
    test_value = test_value.cpu()
    test_output = test_output.cpu()
    print(loss, torch.sum(torch.abs(test_output - test_value) / (test_value + 1e-9)) / len(test_output))