# %%
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
def pathological_curve_loss(x: Tensor, y: Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss, min_points=[(0, "y_min")])

# %%
def our_opt_fn_with_sgd(
    fn: Callable, xy: Float[Tensor, "2"], lr=0.005, momentum=0.98, n_iters: int = 100
) -> Float[Tensor, "n_iters_plus_1 2"]:
    """
    Optimize a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters+1, 2). The (x, y) values, from initial values to values after step `n_iters`.
    """
    # Make sure tensor has requires_grad=True, otherwise it can't be optimized (more on this tomorrow!)
    assert xy.requires_grad
    xy_list = [xy]

    for _ in range(n_iters):
        xy.retain_grad()
        loss = fn(x=xy[0], y=xy[1])
        loss.backward()
        print(f"xy: {xy}")
        print(f"loss: {loss}")
        xy_grad = xy.grad
        print(f"gradient: {xy_grad}")
        update = lr * xy_grad
        print(f'Update: {update}')
        xy = xy - update
        xy_list.append(xy.detach().clone())
        xy.grad = None

    print(t.stack(xy_list).shape)
    print(xy_list)
    return t.stack(xy_list).detach()




points = []

optimizer_list = [
    (optim.SGD, {"lr": 1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.2, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = our_opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])

# %%
def opt_fn_with_sgd(
    fn: Callable, xy: Float[Tensor, "2"], lr=0.001, momentum=0.98, n_iters: int = 100
) -> Float[Tensor, "n_iters_plus_1 2"]:
    """
    Optimize a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters+1, 2). The (x, y) values, from initial values to values after step `n_iters`.
    """
    # Make sure tensor has requires_grad=True, otherwise it can't be optimized (more on this tomorrow!)
    assert xy.requires_grad

    optimizer = optim.SGD((xy,), lr=lr, momentum=momentum)

    xy_list = [xy.detach().clone()]  # so we don't unintentionally modify past values in `xy_list`

    for i in range(n_iters):
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        optimizer.zero_grad()
        xy_list.append(xy.detach().clone())

    return t.stack(xy_list)


points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])

# %%
class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        self.params = list(params)  # turn params into a list (it might be a generator, so iterating over it empties it)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        self.b = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm."""
        #print(f"Params List: {[p for p in self.params]}")
        g_t = [p.grad for p in self.params]
        # print(self.params[0].shape)
        # print(self.params[1].shape)
        # print(self.params[2].shape)
        # print(g_t[0].shape)
        # print(g_t[1].shape)
        # print(g_t[2].shape)
        for i, param in enumerate(self.params):
            if self.lmda != 0.0:
                g_t[i] += self.lmda * self.params[i]
            if self.mu != 0.0:
                self.b[i] = self.mu * self.b[i] + g_t[i]
                g_t[i].copy_(self.b[i])
            self.params[i] -= self.lr * g_t[i]
        


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)

# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
        """
        self.params = list(params)  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.b = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        g_t = [p.grad for p in self.params]
        for i, param in enumerate(self.params):
            if self.lmda != 0.0:
                g_t[i] += self.lmda * self.params[i]
            self.v[i] = self.alpha * self.v[i] + (1- self.alpha) * g_t[i]**2
            g_t[i] = g_t[i] / (t.sqrt(self.v[i]) + self.eps)
            if self.mu != 0.0:
                self.b[i] = self.mu * self.b[i] + g_t[i]
                g_t[i].copy_(self.b[i])
            self.params[i] -= self.lr * g_t[i]

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"
        )


tests.test_rmsprop(RMSprop)

# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        g_t = [p.grad for p in self.params]
        for i, param in enumerate(self.params):
            if self.lmda != 0.0:
                g_t[i] += self.lmda * self.params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t[i]
            self.v[i] = self.beta2 * self.v[i] + (1- self.beta2) * g_t[i]**2
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
            self.params[i] -= self.lr * m_hat / (t.sqrt(v_hat) + self.eps)
        self.t +=1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)

# %%
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements AdamW.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g_t = [p.grad for p in self.params]
            self.params[i] -= self.lr * self.lmda * self.params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t[i]
            self.v[i] = self.beta2 * self.v[i] + (1- self.beta2) * g_t[i]**2
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
            self.params[i] -= self.lr * m_hat / (t.sqrt(v_hat) + self.eps)
        self.t +=1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"

tests.test_adamw(AdamW)

# %%
def opt_fn(
    fn: Callable,
    xy: Tensor,
    optimizer_class,
    optimizer_hyperparams: dict,
    n_iters: int = 100,
) -> Tensor:
    """Optimize a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, Adam or AdamW
    optimizer_hyperparams: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    """
    assert xy.requires_grad

    optimizer = optimizer_class([xy], **optimizer_hyperparams)

    xy_list = [xy.detach().clone()]  # so that we don't unintentionally modify past values in `xy_list`

    for i in range(n_iters):
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        optimizer.zero_grad()
        xy_list.append(xy.detach().clone())

    return t.stack(xy_list)


points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    (AdamW, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(
        pathological_curve_loss,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(pathological_curve_loss, min_points=[(0, "y_min")], points=points)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cifar_trainset = CIFAR10(train=True)
cifar_testset = CIFAR10(train=False)

imshow(
    cifar_trainset.data[:15],
    facet_col=0,
    facet_col_wrap=5,
    facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
    title="CIFAR-10 images",
    height=600,
    width=1000,
)

# %%
@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class ResNetFinetuner:
    def __init__(self, args: ResNetFinetuningArgs):
        self.args = args

    def pre_training_setup(self):
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = AdamW(
            self.model.out_layers[-1].parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )


        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""

        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        self.logged_variables["accuracy"].append(accuracy)
        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    ex_seen=f"{self.examples_seen:06}",
                    refresh=False,
                )

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.2f}",
                ex_seen=f"{self.examples_seen:06}",
            )

        return self.logged_variables

# %%
args = ResNetFinetuningArgs()
trainer = ResNetFinetuner(args)
logged_variables = trainer.train()
# %%
line(
    y=[logged_variables["loss"][: 391 * 3 + 1], logged_variables["accuracy"][:4]],
    x_max=len(logged_variables["loss"][: 391 * 3 + 1] * args.batch_size),
    yaxis2_range=[0, 1],
    use_secondary_yaxis=True,
    labels={"x": "Examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="Feature extraction with ResNet34",
    width=800,
)
# %%
def test_resnet_on_random_input(model: ResNet34, n_inputs: int = 3, seed: int | None = 42):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    device = next(model.parameters()).device
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: Tensor = model(x.to(device))
    probs = logits.softmax(-1)
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(img, width=200, height=200, margin=0, xaxis_visible=False, yaxis_visible=False)
        bar(
            prob,
            x=cifar_trainset.classes,
            width=600,
            height=400,
            text_auto=".2f",
            labels={"x": "Class", "y": "Prob"},
        )


test_resnet_on_random_input(trainer.model)
# %%
@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False


class WandbResNetFinetuner(ResNetFinetuner):
    args: WandbResNetFinetuningArgs  # adding this line helps with typechecker!
    examples_seen: int = 0  # tracking examples seen (used as step for wandb)

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`."""
        super().pre_training_setup()
        wandb.init()

    # def training_step(
    #     self,
    #     imgs: Float[Tensor, "batch channels height width"],
    #     labels: Int[Tensor, " batch"],
    # ) -> Float[Tensor, ""]:
    #     """Equivalent to ResNetFinetuner.training_step, but logging the loss to wandb."""
    #     raise NotImplementedError()
    
    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""

        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        wandb.log({"loss": loss.item()}, self.examples_seen)
        return loss

    # @t.inference_mode()
    # def evaluate(self) -> float:
    #     """Equivalent to ResNetFinetuner.evaluate, but logging the accuracy to wandb."""
    #     raise NotImplementedError()

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        wandb.log({"accuracy": accuracy}, self.examples_seen)
        return accuracy

    # def train(self) -> None:
    #     """Equivalent to ResNetFinetuner.train, but with wandb integration."""
    #     self.pre_training_setup()
    #     raise NotImplementedError()

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    ex_seen=f"{self.examples_seen:06}",
                    refresh=False,
                )

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.2f}",
                ex_seen=f"{self.examples_seen:06}",
            )
        wandb.finish()
        return self.logged_variables


args = WandbResNetFinetuningArgs(use_wandb=True)
trainer = WandbResNetFinetuner(args)
trainer.train()

# %%
# YOUR CODE HERE - fill `sweep_config` so it has the requested behaviour
sweep_config = dict(
    method = "random",
    metric = dict(
        name = "accuracy", # name of the metric you're optimising (should be a numeric type logged in `wandb.log`)
        goal = "maximize", # either "maximize" or "minimize"
    ),
    parameters = dict(
        lr = dict(min=1e-4, max=1e-1, distribution="log_uniform_values"),
        batch_size = dict(values = [32, 64, 128, 256]),
        weight_decay = dict(min=1e-4, max=1e-2, distribution="log_uniform_values"),
        use_weight_decay = dict(values = [True, False]),
    ),
)


def update_args(args: WandbResNetFinetuningArgs, sampled_parameters: dict) -> WandbResNetFinetuningArgs:
    """
    Returns a new args object with modified values. The dictionary `sampled_parameters` will have
    the same keys as your `sweep_config["parameters"]` dict, and values equal to the sampled values
    of those hyperparameters.
    """
    print(f"args: {args}")
    args.learning_rate = sampled_parameters["lr"]
    args.batch_size = sampled_parameters["batch_size"]
    args.weight_decay = sampled_parameters["weight_decay"] if sampled_parameters["use_weight_decay"] else 0.0


    return args


tests.test_sweep_config(sweep_config)
tests.test_update_args(update_args, sweep_config)
# %%
def train():
    # Define args & initialize wandb
    args = WandbResNetFinetuningArgs(use_wandb=False)
    wandb.init(project=args.wandb_project, name=args.wandb_name, reinit=False)

    # After initializing wandb, we can update args using `wandb.config`
    args = update_args(args, dict(wandb.config))

    # Train the model with these new hyperparameters (the second `wandb.init` call will be ignored)
    trainer = WandbResNetFinetuner(args)
    trainer.train()


sweep_id = wandb.sweep(sweep=sweep_config, project="day3-resnet-sweep")
wandb.agent(sweep_id=sweep_id, function=train, count=3)
wandb.finish()
# %%
WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    raise NotImplementedError()


if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)
