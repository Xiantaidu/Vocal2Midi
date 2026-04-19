from dataclasses import dataclass
from fnmatch import fnmatch
from functools import partial
from typing import Any, Dict, List, Type, Callable, Optional, Union

import torch
from torch.optim.optimizer import ParamsT

from ..reflection import get_object_by_module_path, filter_kwargs_by_class

__all__ = [
    "ChainedOptimizer",
    "OptimizerSpec",
    "OptimizerPlainSpec",
]


@dataclass
class OptimizerPlainSpec:
    """
    Plain spec for creating an optimizer that is part of a `ChainedOptimizer`,
    able to parse from a configuration file.
    """
    cls: str
    kwargs: Dict[str, Any] = None
    filter: str = None


@dataclass
class OptimizerSpec:
    """Spec for creating an optimizer that is part of a `ChainedOptimizer`."""

    class_type: Type[torch.optim.Optimizer]
    init_args: Dict[str, Any] = None
    param_filter: Optional[Callable[..., bool]] = None


# noinspection PyUnusedLocal
def _default_includes_excludes(
        *,
        name: str, includes: list[str] = None, excludes: list[str] = None,
        **kwargs
) -> bool:
    """
    An includes/excludes-only filter function for the chained optimizer.
    """
    return (
            (not includes or any(fnmatch(name, p) for p in includes)) and
            (not excludes or not any(fnmatch(name, p) for p in excludes))
    )


def _parse_plain_spec(spec: OptimizerPlainSpec) -> OptimizerSpec:
    class_type = get_object_by_module_path(spec.cls)
    init_args = spec.kwargs or {}
    if spec.filter is not None:
        if isinstance(spec.filter, str):
            param_filter = get_object_by_module_path(spec.filter)
        elif isinstance(spec.filter, dict):
            param_filter = partial(
                _default_includes_excludes,
                includes=spec.filter.get("includes"),
                excludes=spec.filter.get("excludes"),
            )
        else:
            raise ValueError(
                f"Invalid filter format: {spec.filter}. Expected either a string referring to a filter function, "
                f"or a dict containing 'includes' and 'excludes' keys."
            )
    else:
        param_filter = None
    return OptimizerSpec(
        class_type=class_type,
        init_args=init_args,
        param_filter=param_filter,
    )


class ChainedOptimizer(torch.optim.Optimizer):
    """
    A wrapper around multiple optimizers that allows for chaining them together.
    The optimizers are applied in the order they are passed in the constructor.
    Each optimizer is responsible for updating a subset of the parameters, which
    is determined by the `param_filter` function. If no optimizer is found for a
    parameter group, an exception is raised.
    """

    def __init__(
            self,
            module_or_params: Union[torch.nn.Module, ParamsT],
            specs: List[Union[OptimizerSpec, OptimizerPlainSpec, Dict[str, Any]]],
            lr: float,
            weight_decay: float = 0.0,
            **common_kwargs,
    ):
        self.parameter_link_to_module: Dict[int, torch.nn.Module] = {}
        self.parameter_link_to_name: Dict[int, str] = {}
        if isinstance(module_or_params, torch.nn.Module):
            params = module_or_params.parameters()
            for module in module_or_params.modules():
                for param in module.parameters(recurse=False):
                    self.parameter_link_to_module[id(param)] = module
            for name, param in module_or_params.named_parameters(recurse=True):
                self.parameter_link_to_name[id(param)] = name
        else:
            params = module_or_params
        optimizer_specs = []
        for i, spec in enumerate(specs):
            if isinstance(spec, OptimizerSpec):
                pass
            elif isinstance(spec, OptimizerPlainSpec):
                spec = _parse_plain_spec(spec)
            elif isinstance(spec, dict):
                spec = _parse_plain_spec(OptimizerPlainSpec(**spec))
            else:
                raise TypeError(
                    f"Element {i} of `specs` must be instance of `OptimizerSpec` or `OptimizerPlainSpec`, "
                    f"or a dict that can be converted to `OptimizerPlainSpec`."
                )
            if i < len(specs) - 1 and spec.param_filter is None:
                raise ValueError(
                    f"Optimizer {i} must have a `param_filter` as it is not the last optimizer in the list."
                )
            optimizer_specs.append(spec)
        self.optimizer_specs = optimizer_specs
        self.optimizers: List[torch.optim.Optimizer] = []
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Split the params for each optimizer
        params_for_optimizers = [[] for _ in optimizer_specs]
        for param_group in self.param_groups:
            params = param_group["params"]
            indices = param_group["optimizer_and_param_group_indices"] = set()
            for param in params:
                assert isinstance(param, torch.Tensor), f"Expected a torch.Tensor, got {type(param)}"
                for index, spec in enumerate(optimizer_specs):
                    if spec.param_filter is None or spec.param_filter(
                            module=self.parameter_link_to_module.get(id(param)),
                            param=param,
                            name=self.parameter_link_to_name.get(id(param)),
                    ):
                        params_for_optimizers[index].append(param)
                        indices.add((index, 0))
                        break

        # Initialize the optimizers
        for spec, selected_params in zip(optimizer_specs, params_for_optimizers):
            optimizer_args = {
                "lr": lr,
                "weight_decay": weight_decay,
            }
            optimizer_args.update(common_kwargs)
            if spec.init_args is not None:
                optimizer_args.update(spec.init_args)
            optimizer = spec.class_type(selected_params, **filter_kwargs_by_class(spec.class_type, optimizer_args))
            self.optimizers.append(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            **super().state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optimizers = state_dict.pop("optimizers")
        super().load_state_dict(state_dict)
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict(optimizers[i])

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def _copy_lr_to_optimizers(self) -> None:
        for param_group in self.param_groups:
            indices = param_group["optimizer_and_param_group_indices"]
            for optimizer_idx, param_group_idx in indices:
                self.optimizers[optimizer_idx].param_groups[param_group_idx]["lr"] = param_group["lr"]

    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._copy_lr_to_optimizers()
        for opt in self.optimizers:
            opt.step(closure=None)
        return loss

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)

        # If optimizer has not been initialized, skip adding the param groups
        if not self.optimizers:
            return

        # Split the params for each optimizer
        params_for_optimizers = [[] for _ in self.optimizer_specs]
        params = param_group["params"]
        indices = param_group["optimizer_and_param_group_indices"] = set()
        for param in params:
            assert isinstance(param, torch.Tensor), f"Expected a torch.Tensor, got {type(param)}"
            found_optimizer = False
            for index, spec in enumerate(self.optimizer_specs):
                if spec.param_filter is None or spec.param_filter(
                        module=self.parameter_link_to_module.get(id(param)),
                        param=param,
                        name=self.parameter_link_to_name.get(id(param)),
                ):
                    params_for_optimizers[index].append(param)
                    indices.add((index, len(self.optimizers[index].param_groups)))
                    found_optimizer = True
                    break
            if not found_optimizer:
                raise ValueError("No valid optimizer found for the given parameter group")

        # Add the selected param group to the optimizers
        for optimizer, selected_params in zip(self.optimizers, params_for_optimizers):
            if selected_params:
                optimizer.add_param_group({"params": selected_params})
