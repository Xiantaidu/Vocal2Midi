import collections

import torch
from torch import nn


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.
    This class maintains a shadow copy of the model parameters and updates them
    using exponential moving average.
    Example::
        model = MyModel()
        ema = ExponentialMovingAverage(
            parameters=dict(model.named_parameters()),
            decay=0.999
        )

        # training loop
        for data in train_dataloader:
            loss = model(data)
            ...
            optimizer.step()
            ema.step()  # update the shadow parameters

        # validation loop
        ema.apply()  # switch model to shadow parameters
        for data in val_dataloader:
            val_loss = model(data)
            ...
        ema.restore()  # restore model to original parameters

        # save the parameters
        torch.save(model.state_dict(), "model.pth")
        torch.save(ema.state_dict(), "ema.pth")
    :param parameters: Dictionary of model parameters.
    :param decay: Decay rate for the moving average.
    """
    def __init__(self, parameters: dict[str, nn.Parameter], decay=0.99):
        self.decay = decay
        self.referenced: dict[str, nn.Parameter] = {
            name: param
            for name, param in parameters.items()
            if param.requires_grad
        }
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.register()

    def size(self) -> int:
        """
        Returns the number of shadow parameters.
        """
        return len(self.shadow)

    def register(self):
        """
        Manually copy the referenced parameters to the shadow parameters.
        """
        self.shadow.clear()
        for name, param in self.referenced.items():
            self.shadow[name] = param.data.clone()

    def step(self):
        """
        Perform a single step to update the shadow parameters.
        """
        for name, param in self.referenced.items():
            new_param = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
            self.shadow[name] = new_param.clone()

    def apply(self):
        """
        Backup the original parameters and replace them with the shadow parameters.
        """
        for name, param in self.referenced.items():
            self.backup[name] = param.data
            param.data = self.shadow[name].to(param.device)

    def restore(self):
        """
        Restore the original parameters from the backup.
        """
        for name, param in self.referenced.items():
            param.data = self.backup[name].to(param.device)
        self.backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns the state dictionary containing the shadow parameters.
        """
        return collections.OrderedDict((name, self.shadow[name].clone()) for name in sorted(self.shadow.keys()))

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        """
        Load the shadow parameters from a state dictionary.
        """
        if strict:
            source_keys = set(state_dict.keys())
            target_keys = set(self.shadow.keys())
            missing_keys = target_keys - source_keys
            unexpected_keys = source_keys - target_keys
            if missing_keys:
                raise KeyError(
                    f"Missing keys in state_dict:\n"
                    + "\n".join(f"  {key}" for key in missing_keys)
                )
            if unexpected_keys:
                raise KeyError(
                    f"Unexpected keys in state_dict:\n"
                    + "\n".join(f"  {key}" for key in unexpected_keys)
                )
        for name, tensor in self.shadow.items():
            if tensor.shape != state_dict[name].shape:
                raise RuntimeError(
                    f"Shape mismatch for key '{name}': "
                    f"expected {tensor.shape}, got {state_dict[name].shape}"
                )
            self.shadow[name] = state_dict[name].clone().to(tensor.device)
