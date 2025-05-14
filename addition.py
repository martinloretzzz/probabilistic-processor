import os
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset
import matplotlib.pyplot as plt
import wandb

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

        std = 0.05
        torch.nn.init.normal_(self.wvu.weight, mean=0.0, std=std)
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

        scale = 0.0026034886748242643
        self.program = nn.Parameter(torch.randn(config.program_length, config.inst_width) * scale)
        # torch.nn.init.normal_(self.program, mean=0.0, std=0.002)
        self.loop_counter = nn.Parameter(torch.empty(self.loop_count, config.hidden_size))
        self.instruction_counter = nn.Parameter(torch.empty(self.program_length, config.hidden_size))
        torch.nn.init.normal_(self.loop_counter, mean=0.0, std=0.2)
        torch.nn.init.normal_(self.instruction_counter, mean=0.0, std=0.2)

        self.alu = ProcessorUnit(config)
        self.ln = nn.LayerNorm(config.hidden_size)

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
            loop_i = self.loop_counter[j]
            for i in range(self.program_length):
                inst_i = self.instruction_counter[i]
                inst = self.program[i]
                inst_pos = (loop_i + inst_i).unsqueeze(0)
                x = x + self.alu(self.ln(x + inst_pos), inst)
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

def test(model, device, test_loader, epoch):
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
    test_accuracy = 100. * correct / len(test_loader.dataset)

    wandb.log({"epoch": epoch, "test_loss": test_loss, "test_accuracy": test_accuracy})

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy

def train(model, device, train_loader, optimizer, scheduler, epoch, test_loader, log_interval=10, test_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        x1, x2 = torch.split(data, split_size_or_sections=1, dim=-1)
        x1, x2, target = x1.squeeze().to(device), x2.squeeze().to(device), target.to(device)

        optimizer.zero_grad()
        output, loss = model(x1, x2, targets=target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            wandb.log({"epoch": epoch, "batch_idx": batch_idx, "train_loss": loss.item(), "lr": lr, "norm": norm})

        if batch_idx % test_interval == 0:
            test(model, device, test_loader, epoch)

def plot(model, device, size, file):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    model.eval()
    test_loss, correct = 0, 0
    encoder = NumberEncoder(model.config.digits)
    x = torch.linspace(0, size, steps=size, dtype=torch.long)
    x1 = encoder.encode(x).to(device)
    heatmap = []
    with torch.no_grad():
        for y in range(size):
            x2 = encoder.encode(torch.tensor(y).repeat(size)).to(device)
            z = (x + y).to(device)
            output, loss = model(x1, x2, encoder.encode(z))
            test_loss += loss.item()
            line = (encoder.decode_onehot(output) == z)
            correct += line.sum().item()
            heatmap.append(line)

    heatmap = torch.stack(heatmap)
    test_loss /= size * size
    accuracy = 100. * correct / (size * size)
    print(f'\nTest all numbers {size}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{size * size} ({accuracy:.0f}%)\n')

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.cpu().numpy(), origin='lower', extent=[0, size, 0, size], cmap='binary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(file)

    wandb.log({
        "plot_test_loss": test_loss,
        "plot_accuracy": accuracy,
        "heatmap_size": size,
        "heatmap": wandb.Image(file)
    })
    plt.close()

train_dataset_size = 20000
test_dataset_size = 10000
batch_size = 256
test_batch_size = 10000
epochs = 32
seed = 1

max_lr = 3
min_lr = max_lr * 0.1
max_steps = epochs * (train_dataset_size // batch_size)
warmup_steps = 2 * (train_dataset_size // batch_size)

Config = namedtuple('Config', [
    'digits', 'program_length', 'loop_count', 'inst_width',
    'hidden_size', 'core_size', 'num_cores'
])

config = Config(
    digits=6,
    program_length=2,
    loop_count=6,
    inst_width=128,
    hidden_size=128,
    core_size=64,
    num_cores=32
)

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(gen_dataset(digits=config.digits, dataset_size=train_dataset_size), shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(gen_dataset(digits=config.digits, dataset_size=test_dataset_size), shuffle=False, batch_size=test_batch_size)

model = BinaryOperationNet(config).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=max_lr)

total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"Total parameter count: {total_params}")

wandb.init(
    project="probabilistic-processor",
    name=f"b-d{config.digits}-l{config.program_length}",
    config={
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "epochs": epochs,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "seed": seed,
        "total_parameters": total_params,
        "train_dataset_size": train_dataset_size,
        "test_dataset_size": test_dataset_size,
        **config._asdict()
    }
)

cosine_steps = max_steps - warmup_steps
warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, scheduler, epoch, test_loader, log_interval=10, test_interval=100)
    # plot(model, device, size=100, file="./out/add-100.png")

test(model, device, test_loader, epoch=epochs)
plot(model, device, size=1000, file="./out/add-1000.png")

wandb.finish()
print("Done")