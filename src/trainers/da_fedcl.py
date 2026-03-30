from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch


class DAFedCLTrainer(BaseTrainer):
    """DA-FedCL trainer with client-specific short/long memories.

    Local objective:
        L = L_task
          + mu * ||w - w_global||^2
          + lambda_s * ||w - w_prev||^2
          + lambda_l * ||w - w_ema||^2

    Notes:
    - Algorithm logic is kept in trainer layer.
    - Worker is only used as a generic compute engine with loss_hook interface.
    """

    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(DAFedCLTrainer, self).__init__(options, dataset, model, self.optimizer)

        self.mu = float(options.get('mu', 0.0))
        self.lambda_s = float(options.get('lambda_s', 0.0))
        self.lambda_l = float(options.get('lambda_l', 0.0))
        self.alpha = float(options.get('alpha', 0.9))

        # client-level memories (initialized at train start after latest_model is determined)
        self.client_prev = {}
        self.client_ema = {}

    @staticmethod
    def _flat_params_with_grad(model):
        return torch.cat([p.view(-1) for p in model.parameters()])

    def _initialize_client_memories(self):
        init_flat = self.latest_model.detach().clone()
        self.client_prev = {c.cid: init_flat.clone() for c in self.clients}
        self.client_ema = {c.cid: init_flat.clone() for c in self.clients}

    def _build_loss_hook(self, global_model, prev_anchor, ema_anchor):
        printed_once = {'flag': False}

        def _hook(worker, base_loss):
            current = self._flat_params_with_grad(worker.model)
            reg = 0.0

            if self.mu > 0.0:
                g = global_model
                if g.device != current.device:
                    g = g.to(current.device)
                reg = reg + self.mu * torch.sum((current - g) ** 2)

            if self.lambda_s > 0.0:
                p = prev_anchor
                if p.device != current.device:
                    p = p.to(current.device)
                reg = reg + self.lambda_s * torch.sum((current - p) ** 2)

            if self.lambda_l > 0.0:
                e = ema_anchor
                if e.device != current.device:
                    e = e.to(current.device)
                reg = reg + self.lambda_l * torch.sum((current - e) ** 2)

            if self.print_result and (not printed_once['flag']):
                try:
                    reg_val = float(reg.item()) if hasattr(reg, 'item') else float(reg)
                    print(f'>>> DA-FedCL reg term: {reg_val:.6f} (mu={self.mu}, lambda_s={self.lambda_s}, lambda_l={self.lambda_l})')
                except Exception:
                    pass
                printed_once['flag'] = True

            return base_loss + reg

        return _hook

    def _update_client_memories(self, cid, local_solution):
        local_solution = local_solution.detach().clone()
        self.client_prev[cid] = local_solution.clone()

        old_ema = self.client_ema.get(cid, local_solution.clone())
        if old_ema.device != local_solution.device:
            old_ema = old_ema.to(local_solution.device)
        new_ema = self.alpha * old_ema + (1.0 - self.alpha) * local_solution
        self.client_ema[cid] = new_ema.detach().clone()

    def local_train(self, round_i, selected_clients, **kwargs):
        solns = []
        stats = []

        # snapshot current global model as a fixed anchor for this round
        global_anchor = self.latest_model.detach().clone()

        for i, c in enumerate(selected_clients, start=1):
            c.set_flat_model_params(self.latest_model)

            prev_anchor = self.client_prev.get(c.cid, global_anchor).detach().clone()
            ema_anchor = self.client_ema.get(c.cid, global_anchor).detach().clone()
            loss_hook = self._build_loss_hook(global_anchor, prev_anchor, ema_anchor)

            soln, stat = c.local_train(loss_hook=loss_hook)

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                          round_i, c.cid, i, self.clients_per_round,
                          stat['norm'], stat['min'], stat['max'],
                          stat['loss'], stat['acc'] * 100, stat['time']))

            solns.append(soln)
            stats.append(stat)

            _, local_solution = soln
            self._update_client_memories(c.cid, local_solution)

        return solns, stats

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        self.latest_model = self.worker.get_flat_model_params().detach()
        self._initialize_client_memories()

        for round_i in range(self.num_round):
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            selected_clients = self.select_clients(seed=round_i)
            solns, stats = self.local_train(round_i, selected_clients)

            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)

            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)
        self.metrics.write()
