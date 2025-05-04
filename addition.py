import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset


class NumberEmbedder(nn.Module):
    def __init__(self, digits=4, hidden_size=128):
        super(NumberEmbedder, self).__init__()
        self.digits, self.hidden_size = digits, hidden_size
        self.emb = nn.Embedding(digits * 10, hidden_size)

    def forward(self, nums):
        if any([x for x in nums if x > pow(10, self.digits)]):
            raise Exception(f"Can't embed numbers with more than {self.digits}")
        device = self.emb.weight.device

        digit_arr = self.to_digit_tensor(nums)
        out = []
        for num_digits in digit_arr:
            y = torch.zeros((self.hidden_size,), device=device)
            for i, d in enumerate(num_digits):
                y = y + self.emb(torch.tensor(10 * i + d, dtype=torch.long, device=device))
            out.append(y)
        return torch.stack(out)

    def forward_output(self, x):
        return x @ self.emb.weight.T

    def decode_numbers(self, x):
        digits = x.reshape(-1, self.digits, 10).argmax(-1)
        bases = torch.tensor([pow(10, i) for i in range(self.digits)], device=x.device).reshape(1,self.digits)
        return (digits * bases).sum(-1)

    def to_digit_tensor(self, nums):
        return [list(reversed([int(x) for x in str(num).zfill(self.digits)])) for num in nums]


class ProcessorNet(nn.Module):
    def __init__(self, program_length=0, inst_width=16, hidden_size=128):
        super(ProcessorNet, self).__init__()
        self.program_length = program_length
        self.inst_width = inst_width

        scale = 1.0 / (0.1 + program_length * inst_width)
        self.program = nn.Parameter(torch.randn(program_length, inst_width) * scale)

        self.alu_down = nn.Linear(hidden_size, inst_width)
        self.alu_up = nn.Linear(inst_width, hidden_size)

    def forward(self, x):
        for i in range(self.program_length):
            inst = self.program[i]
            reg = self.alu_down(x)
            reg = F.relu(reg)
            reg = reg * inst
            x = x + self.alu_up(reg)
        return x


class BinaryOperationNet(nn.Module):
    def __init__(self, program_length=16, inst_width=16, hidden_size=128, digits=4):
        super(BinaryOperationNet, self).__init__()
        self.processor = ProcessorNet(program_length, inst_width, hidden_size)

        self.embedder = NumberEmbedder(digits, hidden_size=hidden_size // 2)

    def forward(self, x1, x2, targets=None):
        x1 = self.embedder(x1)
        x2 = self.embedder(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.processor(x)
        x = x[:, :self.embedder.hidden_size]
        out = self.embedder.forward_output(x)

        loss = None
        if targets is not None:
            target_nums = torch.tensor(self.embedder.to_digit_tensor(targets), device=x.device)
            loss = F.cross_entropy(out.view(-1, 10), target_nums.view(-1))
        return out, loss

def gen_dataset(dataset_size = 10000, max_value = 10000-2):
    x1 = torch.randint(low=0, high=max_value // 2, size=(dataset_size,))
    x2 = torch.randint(low=0, high=max_value // 2, size=(dataset_size,))
    x = torch.stack((x1, x2), dim=-1)
    y = x1 + x2
    return TensorDataset(x, y)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            x1, x2 = torch.split(data, split_size_or_sections=1, dim=1)
            x1, x2, target = x1.squeeze().tolist(), x2.squeeze().tolist(), target.tolist()

            output, loss = model(x1, x2, targets=target)
            test_loss += loss.item()
            correct += (model.embedder.decode_numbers(output) == torch.tensor(target, dtype=torch.long, device=output.device)).sum(-1).item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')    


def train(model, device, train_loader, optimizer, epoch, test_loader, log_interval=10, test_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        x1, x2 = torch.split(data, split_size_or_sections=1, dim=1)
        x1, x2, target = x1.squeeze().tolist(), x2.squeeze().tolist(), target.tolist()

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
epochs = 4
lr = 0.5
gamma = 0.7
seed = 1

torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(gen_dataset(dataset_size=100000, max_value=998), shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(gen_dataset(dataset_size=1000, max_value=998), shuffle=False, batch_size=test_batch_size)

model = BinaryOperationNet(digits=3, program_length=32, inst_width=32, hidden_size=32).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, test_loader, log_interval=10, test_interval=100)
    scheduler.step()
