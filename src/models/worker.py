from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch


criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.options = options
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options["model"] == '2nn' or options["model"] == 'logistic':
            self.flat_data = True
        else:
            self.flat_data = False

        # Setup local model and evaluate its statics
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8
    
    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def _apply_task_aware_logits_labels(self, pred, y):
        """Apply task-aware logit slicing and label remapping if configured.

        Uses:
            self.options['active_labels']: list of global labels for current task
            self.options['label_map']: dict global_label -> local_label
        """
        active_labels = self.options.get('active_labels', None)
        if active_labels is None or len(active_labels) == 0:
            return pred, y

        # Slice logits to active task label space
        idx = torch.tensor(active_labels, device=pred.device, dtype=torch.long)
        pred_sliced = pred.index_select(1, idx)

        # Remap global labels to local labels
        label_map = self.options.get('label_map', None)
        if label_map is None:
            label_map = {int(g): int(i) for i, g in enumerate(active_labels)}

        y_cpu = y.detach().cpu().numpy().tolist()
        y_local_list = [label_map[int(v)] for v in y_cpu]
        y_local = torch.tensor(y_local_list, device=pred.device, dtype=torch.long)
        return pred_sliced, y_local

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:            
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            pred, y = self._apply_task_aware_logits_labels(pred, y)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        # Optional old-task anchor (sequential CL): L_task + lambda_old * ||w - w_old||^2
        prev_model = self.options.get('prev_model', None)
        lambda_old = float(self.options.get('lambda_old', 0.0))
        printed_reg_once = False
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)
                pred_eval = pred
                pred, y = self._apply_task_aware_logits_labels(pred, y)

                if torch.isnan(pred_eval.max()):
                    raise RuntimeError(
                        "RuntimeError[Worker.local_train]: NaN detected in model output logits."
                    )

                loss = criterion(pred, y)

                # Historical parameter regularization to previous-task global model
                # Task-1 has no previous model; task2/task3 use previous task snapshot as anchor.
                # Uses unnormalized squared L2 penalty: torch.sum((w - w_old)**2)
                if prev_model is not None and lambda_old > 0.0:
                    try:
                        current_flat = torch.cat([p.view(-1) for p in self.model.parameters()])
                        prev_flat = prev_model.detach()
                        if prev_flat.device != current_flat.device:
                            prev_flat = prev_flat.to(current_flat.device)
                        reg_loss_old = torch.sum((current_flat - prev_flat) ** 2)
                        loss = loss + lambda_old * reg_loss_old
                        if not printed_reg_once:
                            print(f'>>> old-model reg enabled: lambda_old={lambda_old}, reg_loss={reg_loss_old.item():.6f}')
                            printed_reg_once = True
                    except Exception as e:
                        # Keep baseline behavior if shape/device alignment fails
                        print(f"Warning[Worker.local_train]: old-model regularization skipped due to {type(e).__name__}: {e}")

                # FedProx proximal term: add (mu/2) * ||w - w_global||^2 to loss
                # Uses unnormalized squared L2 penalty: torch.sum((w - w_global)**2)
                prox_mu = kwargs.get('prox_mu', 0.0)
                if prox_mu and prox_mu > 0.0 and 'global_params' in kwargs:
                    global_params = kwargs['global_params']
                    # Build a flat tensor of current parameters that supports autograd
                    current_flat = torch.cat([p.view(-1) for p in self.model.parameters()])
                    if self.gpu:
                        global_params = global_params.cuda()
                        current_flat = current_flat.cuda()
                    # make sure tensors align
                    try:
                        prox = 0.5 * prox_mu * torch.sum((current_flat - global_params) ** 2)
                        loss = loss + prox
                    except Exception as e:
                        # If shapes mismatch, skip proximal term (user should ensure correct shapes)
                        print(f"Warning[Worker.local_train]: FedProx proximal term skipped due to {type(e).__name__}: {e}")

                # Generic algorithm hook: trainer can inject extra regularization terms here
                loss_hook = kwargs.get('loss_hook', None)
                if callable(loss_hook):
                    loss = loss_hook(self, loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss/train_total,
                       "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                pred, y = self._apply_task_aware_logits_labels(pred, y)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch*10):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict


class LrAdjustWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrAdjustWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        m = kwargs['multiplier']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)
        
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch*10):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        
        self.optimizer.set_lr(current_lr)
        return local_solution, return_dict
