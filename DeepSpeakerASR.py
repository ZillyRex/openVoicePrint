import math
import os

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from python_speech_features import mfcc


"""
特征提取
"""
# ！！！每个说话人仅使用前'MAX_SAMPLES'条训练样本，当'MAX_SAMPLES'取sys.maxsize时使用所有训练样本
MAX_SAMPLES = 30


# ！！可以尝试其他特征，比如FBank，或者增加一阶MFCC和二阶MFCC
def make_feature(wav_path, feature_path):
    if not os.path.exists(feature_path):
        rate, sig = wav.read(wav_path)
        feature = mfcc(sig, rate)
        np.save(feature_path, feature)
    return


train_wav_dir = '/mnt/datasets/tongdun_competition/1st_round/training_set/'
test_wav_dir = '/mnt/datasets/tongdun_competition/1st_round/test_set'
train_feature_dir = '/home/kesci/work/train_set/mfcc/'
test_feature_dir = '/home/kesci/work/test_set/mfcc/'
if not os.path.exists(train_feature_dir):
    os.makedirs(train_feature_dir)
if not os.path.exists(test_feature_dir):
    os.makedirs(test_feature_dir)

# train dataset
for speaker in os.listdir(train_wav_dir):
    train_wav_subdir = os.path.join(train_wav_dir, speaker)
    if os.path.isdir(train_wav_subdir):
        train_feature_subdir = os.path.join(train_feature_dir, speaker)
        if not os.path.exists(train_feature_subdir):
            os.makedirs(train_feature_subdir)
        samples_n = 0
        for wav_file in os.listdir(train_wav_subdir):
            if wav_file[0] != '.':
                samples_n += 1
                if samples_n > MAX_SAMPLES:
                    break
                make_feature(os.path.join(train_wav_subdir, wav_file),
                             os.path.join(train_feature_subdir, wav_file).replace('.wav', '.npy'))

# test dataset
for wav_file in os.listdir(test_wav_dir):
    if wav_file[0] != '.':
        make_feature(os.path.join(test_wav_dir, wav_file),
                     os.path.join(test_feature_dir, wav_file).replace('.wav', '.npy'))


"""
数据加载
"""
# ！每次随机选取的特征段长度
NUM_PREVIOUS_FRAME = 9
NUM_NEXT_FRAME = 23
NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME

FILTER_BANK = 64


class TruncatedInputFromMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, input_per_file=1):

        super(TruncatedInputFromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        num_frames = len(frames_features)
        import random

        for i in range(self.input_per_file):

            j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
            if not j:
                frames_slice = np.zeros(NUM_FRAMES, FILTER_BANK, 'float64')
                frames_slice[0:frames_features.shape[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)


class ToTensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            return img


pair_txt = '/mnt/datasets/tongdun_competition/1st_round/pair_id.txt'


class SpeakerDataset(Dataset):
    def __init__(self, num_classes=None, transform=None):
        self.training = (num_classes is None)
        self.transform = transform
        self.features = []
        if self.training:
            self.classes = []
            self.num_classes = 0
            self.class_id_table = []
            for speaker in os.listdir(train_feature_dir):
                self.class_id_table.append(speaker)
                train_feature_subdir = os.path.join(train_feature_dir, speaker)
                if os.path.isdir(train_feature_subdir):
                    for feature_file in os.listdir(train_feature_subdir):
                        if feature_file[0] != '.':
                            self.features.append(os.path.join(train_feature_subdir, feature_file))
                            self.classes.append(self.num_classes)
                self.num_classes += 1
        else:
            self.pairID = []
            xxx=0
            with open(pair_txt) as f:
                pairs = f.readlines()
                for pair in pairs:
                    xxx+=1
                    if(xxx==1):
                        continue
                    self.pairID.append(pair.strip())
                    pair = pair.split('_')
                    self.features.append((os.path.join(test_feature_dir, '{}.npy'.format(pair[0].strip())),
                                          os.path.join(test_feature_dir, '{}.npy'.format(pair[1].strip()))))

    def __getitem__(self, index):
        if self.training:
            feature = self.transform(np.load(self.features[index]))
            return feature, self.classes[index]
        else:
            return self.pairID[index],\
                   self.transform(np.load(self.features[index][0])), self.transform(np.load(self.features[index][1]))

    def __len__(self):
        return len(self.features)


"""
模型
"""


class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ！可以尝试修改网络结构
class ResNet(nn.Module):
    def __init__(self, layers, block=BasicBlock, num_classes=1000):
        super(ResNet, self).__init__()

        self.relu = ReLU(inplace=True)

        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.in_channels = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])

        self.in_channels = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])

        self.in_channels = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avg_pool = nn.AdaptiveAvgPool2d([4, 1])
        self.fc = nn.Linear(512 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


"""
设置参数
"""


# ！！！可以尝试不同的训练参数
class Args:
    def __init__(self):
        # model options
        self.num_classes = 1000  # number of speakers
        self.batch_size = 512  # input batch size for training (default: 128)
        self.test_batch_size = 64  # input batch size for testing (default: 64)
        self.lr = 0.9  # learning rate (default: 0.125)
        self.lr_decay = 1e-4  # learning rate decay ratio (default: 1e-4)
        self.wd = 0.0  # weight decay (default: 0.0)
        # training options
        self.seed = 0  # random seed (default: 0)
        self.model_dir = '/home/kesci/work/model/'
        self.resume = self.model_dir + 'net.pth'  # path to checkpoint model (default: None)
        self.start_epoch = 1  # manual epoch number (useful on restarts)
        self.epochs = 1  # number of epochs to train (default: 10)


args = Args()
np.random.seed(args.seed)
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)


"""
训练/预测
"""


def main():
    print("Begin")
    model = ResNet(layers=[1, 1, 1, 1], num_classes=args.num_classes)
    # ！！可以尝试不同的优化器及参数
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, dampening=0.9, weight_decay=args.wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.wd)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.8, eps=1e-06, weight_decay=args.wd)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    transform = transforms.Compose([
        TruncatedInputFromMFB(),
        ToTensor()
    ])
    train_dataset = SpeakerDataset(transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    start = args.start_epoch
    end = args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)

    transform_test = transforms.Compose([
        TruncatedInputFromMFB(),
        ToTensor()
    ])
    test_dataset = SpeakerDataset(num_classes=args.num_classes, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    test(test_loader, model)
    print("End")


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    # ！！！可以尝试其他损失函数，比如TripletMarginLoss
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.TripletMarginLoss()

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in pbar:
        data, label = Variable(data), Variable(label)

        # compute output
        out = model(data)

        loss = criterion(out, label)
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                loss.data[0]))

    # save model
    # ！！训练应该在什么时候停止?
    #  -> 可以尝试划分一部分训练集作为验证集
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}/net.pth'.format(args.model_dir))


def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    pairs = []
    probs = []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (pair, data_a, data_b) in pbar:
        with torch.no_grad():
            pairs.append(pair)
            data_a, data_b = Variable(data_a), Variable(data_b)

            # compute output
            out_a = model.softmax(model(data_a)).data.numpy()
            out_b = model.softmax(model(data_b)).data.numpy()
            prob = out_a * out_b
            prob = np.sum(prob, axis=1)
            probs.append(prob)

            pbar.set_description('Test  Epoch:     [{}/{} ({:.0f}%)]'.format(
                batch_idx + 1, len(test_loader),
                100. * (batch_idx + 1) / len(test_loader)))

    pairs = np.concatenate(pairs)
    preds = np.concatenate(probs)
    with open('/home/kesci/work/pred.csv', mode='w') as f:
        f.write('pairID,pred\n')
        for i in range(len(preds)):
            f.write('{},{}\n'.format(pairs[i], preds[i])


main()
