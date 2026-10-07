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
def pathological_curve_loss(x: Tensor, y: Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss, min_points=[(0, "y_min")])# %%
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

    # breakpoint()
    xy.detach().clone()
    xy_val_list = [xy.detach().clone()]
    opt = t.optim.SGD(params=(xy,), lr=lr, momentum=momentum)
    for _ in range(n_iters):
        opt.zero_grad()
        loss = fn(xy[0], xy[1])
        loss.backward()
        tens = t.tensor([1.0], requires_grad=True)
        tens.backward()
        # print(tens)
        # print(tens.shape)
        # exit()
        
        opt.step()
        xy_val_list.append(xy.detach().clone())
    return t.stack(xy_val_list)



points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    (optim.SGD, {"lr": 0.5, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])

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
        for i, param in enumerate(self.params):
            gradient = param.grad
            if self.lmda:
                gradient = gradient + self.lmda * param 
            if self.mu:
                self.b[i] = self.mu * self.b[i] + gradient
                gradient = self.b[i]
            param -= self.lr * gradient


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)

import numpy as np
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
        for theta, b, v in zip(self.params, self.b, self.v):
            g = theta.grad
            if self.lmda != 0:
                g = g + self.lmda * theta
            v.copy_(self.alpha * v + (1 - self.alpha) * g.pow(2))  # inplace operation, to modify value in self.v
            g = g / (v.sqrt() + self.eps)  # not inplace operation
            if self.mu > 0:
                b.copy_(self.mu * b + g)  # inplace operation, to modify value in self.b
                g = b
            theta -= self.lr * g  # inplace operation, to modify params


    def __repr__(self) -> str:
        return (f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})")


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
        for i, param in enumerate(self.params):
            gradient = param.grad
            if self.lmda:
                gradient = gradient + self.lmda * param
            if self.beta1:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * gradient * gradient

            m_temp = self.m[i] / (1 - self.beta1** self.t) 
            v_temp = self.v[i] / (1 - self.beta2** self.t) 
            gradient = m_temp / (t.sqrt(v_temp) + self.eps)
            
            param -= self.lr * gradient
        self.t += 1

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
            gradient = param.grad
            if self.lmda:
                param -= self.lmda * param * self.lr
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * gradient * gradient

            m_temp = self.m[i] / (1 - self.beta1** self.t) 
            v_temp = self.v[i] / (1 - self.beta2** self.t) 
            gradient = self.lr * m_temp / (t.sqrt(v_temp) + self.eps)
            
            param -= gradient
        self.t += 1

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
    (SGD, {"lr": 0.02, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.001}),
    (AdamW, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.001}),
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
args = ResNetFinetuningArgs()
trainer = ResNetFinetuner(args)
logged_variables = trainer.train()

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
        wandb.init(
            project = self.args.wandb_project,
            name=self.args.wandb_name,
            config=self.args
        )
        wandb.watch(models=self.model.out_layers[-1], log='all', log_freq=50)

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Equivalent to ResNetFinetuner.training_step, but logging the loss to wandb."""
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
        wandb.log({'loss': loss.item()}, self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
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
        wandb.log({'accuracy': accuracy}, self.examples_seen)
        return accuracy
        

    def train(self) -> None:
        """Equivalent to ResNetFinetuner.train, but with wandb integration."""
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
        

args = WandbResNetFinetuningArgs(use_wandb=False)
trainer = WandbResNetFinetuner(args)
trainer.train()
# %%
# YOUR CODE HERE - fill `sweep_config` so it has the requested behaviour
sweep_config = dict(
    method = "random",
    metric = dict(name="accuracy", goal='maximize'),
    parameters = dict(
        learning_rate=dict(min=1e-4, max=1e-1, distribution='log_uniform_values'),
        batch_size=dict(values=[32,64,128,256]),
        weight_decay=dict(min=1e-4, max=1e-2, distribution='log_uniform_values'),
        weight_decay_bool=dict(values=[True, False]),
    ),
)


def update_args(args: WandbResNetFinetuningArgs, sampled_parameters: dict) -> WandbResNetFinetuningArgs:
    """
    Returns a new args object with modified values. The dictionary `sampled_parameters` will have
    the same keys as your `sweep_config["parameters"]` dict, and values equal to the sampled values
    of those hyperparameters.
    """
    assert set(sampled_parameters.keys()) == set(sweep_config["parameters"].keys())
    args.learning_rate = sampled_parameters['learning_rate']
    args.batch_size = sampled_parameters['batch_size']
    args.weight_decay = sampled_parameters['weight_decay'] if sampled_parameters['weight_decay_bool'] else 0
    return args


tests.test_sweep_config(sweep_config)
tests.test_update_args(update_args, sweep_config)
# %%
