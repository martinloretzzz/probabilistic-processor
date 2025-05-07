from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset

class NumberEncoder():
    def __init__(self, digits):
        self.digits = digits

    def encode(self, nums):
        if torch.any(nums > pow(10, self.digits)):
            raise Exception(f"Can't embed numbers with more than {self.digits}")
        num_list = [list(reversed([int(x) for x in str(num).zfill(self.digits)])) for num in nums.tolist()]
        return torch.tensor(num_list, dtype=torch.long, device=nums.device)

    def decode(self, x, onehot=False):
        assert x.shape[1] == self.digits and len(x.shape) == 2, f"Last dim needs to be of size {self.digits}"
        bases = torch.tensor([pow(10, i) for i in range(self.digits)], device=x.device).reshape(1,self.digits)
        return (x * bases).sum(-1)
    
    def decode_onehot(self, x):
        return self.decode(x.reshape(-1, self.digits, 10).argmax(-1))


class ProcessorUnit(nn.Module):
    def __init__(self, config):
        super(ProcessorUnit, self).__init__()
        self.config = config
        internal_size = config.core_size * config.num_cores

        self.wp = nn.Linear(config.inst_width, internal_size)
        self.wq = nn.Linear(config.hidden_size, internal_size)

        self.wvd = nn.Linear(config.hidden_size, internal_size)
        self.wvu = nn.Linear(internal_size, config.hidden_size)

        n_layer = 8
        torch.nn.init.normal_(self.wvu.weight, mean=0.0, std=0.2 * (2 * n_layer) ** -0.5)
        torch.nn.init.zeros_(self.wvu.bias)
        for layer in [self.wvd, self.wp, self.wq]:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.2)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x, inst):
        BS = x.shape[0]
        xi = self.wp(inst).unsqueeze(0)
        xq = self.wq(x)
        t = F.cosine_similarity(xi.view(1, -1, self.config.core_size), xq.view(BS, -1, self.config.core_size), dim=-1)
        a = F.relu(t).unsqueeze(-1)
        xvd = self.wvd(x).view(BS, -1, self.config.core_size)
        xva = (xvd * a).view(BS, -1)
        out = self.wvu(xva)
        return out


class BinaryOperationNet(nn.Module):
    def __init__(self, config):
        super(BinaryOperationNet, self).__init__()
        self.config = config
        self.emb = nn.Embedding(10 * config.digits, config.hidden_size // 2)
        torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.2)

        self.program_length, self.loop_count = config.program_length, config.loop_count
        self.inst_width = config.inst_width

        scale = 1.0 / (0.1 + config.program_length * config.inst_width)
        self.program = nn.Parameter(torch.randn(config.program_length, config.inst_width) * scale)

        self.alu = ProcessorUnit(config)

    def forward(self, x1, x2, targets=None):
        x1 = self.embed(x1)
        x2 = self.embed(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.forward_processor(x)
        x = x[:, :self.config.hidden_size // 2]
        out = x @ self.emb.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, 10), targets.view(-1))
        return out, loss
    
    def forward_processor(self, x):
        for j in range(self.loop_count):
            for i in range(self.program_length):
                inst = self.program[i]
                x = x + self.alu(x, inst)
        return x

    def embed(self, x):
        offsets = torch.tensor([10 * i for i in range(self.config.digits)], device=x.device).reshape(1, self.config.digits)
        idx = torch.tensor(offsets + x, dtype=torch.long, device=device)
        dig_emb = self.emb(idx)
        return dig_emb.sum(-2)


def gen_dataset(digits, dataset_size, max_value=None):
    if max_value is None:
        max_value = pow(10, digits) - 2
    encoder = NumberEncoder(digits)
    x1 = torch.randint(low=0, high=max_value // 2, size=(dataset_size,))
    x2 = torch.randint(low=0, high=max_value // 2, size=(dataset_size,))

    x = torch.stack((encoder.encode(x1), encoder.encode(x2)), dim=-1)
    y = encoder.encode(x1 + x2)

    return TensorDataset(x, y)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    encoder = NumberEncoder(model.config.digits)
    with torch.no_grad():
        for data, target in test_loader:
            x1, x2 = torch.split(data, split_size_or_sections=1, dim=-1)
            x1, x2, target = x1.squeeze().to(device), x2.squeeze().to(device), target.to(device)

            output, loss = model(x1, x2, targets=target)
            test_loss += loss.item()
            correct += (encoder.decode_onehot(output) == encoder.decode(target)).sum(-1).item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')    


def train(model, device, train_loader, optimizer, epoch, test_loader, log_interval=10, test_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        x1, x2 = torch.split(data, split_size_or_sections=1, dim=-1)
        x1, x2, target = x1.squeeze().to(device), x2.squeeze().to(device), target.to(device)

        optimizer.zero_grad()
        output, loss = model(x1, x2, targets=target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if batch_idx % test_interval == 0:
            test(model, device, test_loader)

batch_size = 256
test_batch_size = 1000
epochs = 16
lr = 2
gamma = 0.8
seed = 1

Config = namedtuple('Config', [
    'digits', 'program_length', 'loop_count', 'inst_width',
    'hidden_size', 'core_size', 'num_cores'
])

config = Config(
    digits=4,
    program_length=3,
    loop_count=4,
    inst_width=128,
    hidden_size=128,
    core_size=64,
    num_cores=8
)

torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(gen_dataset(digits=config.digits, dataset_size=25000), shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(gen_dataset(digits=config.digits, dataset_size=1000), shuffle=False, batch_size=test_batch_size)

model = BinaryOperationNet(config).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

print(f"Total parameter count: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, test_loader, log_interval=10, test_interval=100)
    scheduler.step()
test(model, device, test_loader)
