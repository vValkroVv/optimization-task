import torch

class CounterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        global global_actlog_state
        if global_actlog_state.enable_monitor or global_actlog_state.activation_observer is not None:
            global_actlog_state.layer_idx += 1
        return input

    @staticmethod
    def backward(ctx, grad_output):
        global global_actlog_state
        if global_actlog_state.enable_monitor or global_actlog_state.activation_observer is not None:
            global_actlog_state.layer_idx -= 1
        return grad_output


class GlobalActivationLoggingState:
    def __init__(self):
        self.monitor = None
        self.enable_monitor = False
        self.layer_idx = -1
        self.activation_observer = None

    def set_monitor_variable(self, monitor):
        self.monitor = monitor
    
    def set_enable_monitor_variable(self, enable_monitor):
        self.enable_monitor = enable_monitor

    def set_layer_idx_variable(self, idx):
        self.layer_idx = idx

    def apply_layer_counter(self, input_tensor):
        return CounterFunction.apply(input_tensor)

    def get_enable_monitor_variable(self):
        return self.enable_monitor

    def use_monitor_variable(self, 
                             tensor : torch.Tensor, 
                             name : str, 
                             suffix : str, 
                             layer_depends: bool=True,
                             is_forward: bool=True):
        if self.activation_observer is not None:
            self.activation_observer(tensor, name, suffix, layer_depends, self.layer_idx)

        if not self.enable_monitor or self.monitor.is_forward != is_forward:
            return

        if layer_depends:
            name = name.format(str(self.layer_idx))
        
        self.monitor.add_metrics(self.monitor.metrics, name, suffix, tensor)


class GlobalWeightLoggingState:
    def __init__(self):
        self.monitor = None
        self.enable_monitor = False

    def set_monitor_variable(self, monitor):
        self.monitor = monitor

    def set_enable_monitor_variable(self, enable_monitor):
        self.enable_monitor = enable_monitor

    def get_enable_monitor_variable(self):
        return self.enable_monitor

    def use_monitor_variable(self, tensor: torch.Tensor, name: str, metric_prefix: str, suffix: str = ""):
        if not self.enable_monitor or self.monitor is None:
            return
        self.monitor.add_external_metric(name, tensor, metric_prefix, suffix=suffix)


# Instantiate the GlobalState class once and share it across modules
global_actlog_state = GlobalActivationLoggingState()
global_weightlog_state = GlobalWeightLoggingState()
