from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch
import torch.nn as nn


class FedAvgEWCTrainer(BaseTrainer):
    """FedAvg + EWC baseline for sequential federated continual learning.

    - Task-1 behaves like FedAvg.
    - Task-2+ adds EWC penalty with client-specific cumulative diagonal Fisher.
    """

    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgEWCTrainer, self).__init__(options, dataset, model, self.optimizer)

        self.lambda_ewc = float(options.get('lambda_ewc', 0.0))
        self.ewc_fisher_samples = int(max(0, options.get('ewc_fisher_samples', 128)))
        self.sequential_cl = bool(options.get('sequential_cl', False))

        # Per-client EWC memories transferred across tasks.
        self.client_prev_params = {}
        self.client_fisher_diag = {}

    @staticmethod
    def _flat_params_with_grad(model):
        return torch.cat([p.view(-1) for p in model.parameters()])

    def _build_ewc_loss_hook(self, prev_anchor, fisher_diag):
        printed_once = {'flag': False}

        def _hook(worker, base_loss):
            current = self._flat_params_with_grad(worker.model)

            anchor = prev_anchor
            fisher = fisher_diag
            if anchor.device != current.device:
                anchor = anchor.to(current.device)
            if fisher.device != current.device:
                fisher = fisher.to(current.device)

            reg = 0.5 * self.lambda_ewc * torch.sum(fisher * (current - anchor) ** 2)

            if self.print_result and (not printed_once['flag']):
                try:
                    print(f'>>> FedAvg+EWC reg term: {float(reg.item()):.6f} (lambda_ewc={self.lambda_ewc})')
                except Exception:
                    pass
                printed_once['flag'] = True

            return base_loss + reg

        return _hook

    def _estimate_client_fisher_diag(self, client):
        """Estimate diagonal Fisher on one client using a random subset of train batches."""
        worker = self.worker
        fisher = torch.zeros_like(self.latest_model)
        criterion = nn.CrossEntropyLoss()

        worker.set_flat_model_params(self.latest_model.detach().clone())
        worker.model.train()

        consumed = 0
        used_batches = 0

        for x, y in client.train_dataloader:
            if self.ewc_fisher_samples > 0 and consumed >= self.ewc_fisher_samples:
                break

            if self.ewc_fisher_samples > 0:
                remain = self.ewc_fisher_samples - consumed
                if remain <= 0:
                    break
                if y.size(0) > remain:
                    x = x[:remain]
                    y = y[:remain]

            x = worker.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            worker.optimizer.zero_grad()
            pred = worker.model(x)
            pred, y_local = worker._apply_task_aware_logits_labels(pred, y)
            loss = criterion(pred, y_local)
            loss.backward()

            grad_flat = []
            for p in worker.model.parameters():
                if p.grad is None:
                    grad_flat.append(torch.zeros_like(p).view(-1))
                else:
                    grad_flat.append(p.grad.detach().view(-1))
            g = torch.cat(grad_flat)
            fisher = fisher + g.pow(2)

            consumed += int(y_local.size(0))
            used_batches += 1

        if used_batches > 0:
            fisher = fisher / float(used_batches)

        return fisher.detach().clone()

    def _finalize_ewc_memories(self):
        """Update client Fisher cumulants and parameter anchors at task boundary."""
        if (not self.sequential_cl) or self.lambda_ewc <= 0.0:
            return

        for c in self.clients:
            new_fisher = self._estimate_client_fisher_diag(c)
            old_fisher = self.client_fisher_diag.get(c.cid, None)

            if old_fisher is None:
                cum_fisher = new_fisher
            else:
                if old_fisher.device != new_fisher.device:
                    old_fisher = old_fisher.to(new_fisher.device)
                cum_fisher = old_fisher + new_fisher

            self.client_fisher_diag[c.cid] = cum_fisher.detach().clone()
            self.client_prev_params[c.cid] = self.latest_model.detach().clone()

        print(
            f'>>> FedAvg+EWC memories finalized for {len(self.clients)} clients '
            f'(fisher_samples={self.ewc_fisher_samples}).'
        )

    def export_client_state(self):
        return {
            'client_prev_params': {cid: v.detach().clone() for cid, v in self.client_prev_params.items()},
            'client_fisher_diag': {cid: v.detach().clone() for cid, v in self.client_fisher_diag.items()},
        }

    def import_client_state(self, state):
        if not isinstance(state, dict):
            print('Warning[FedAvgEWCTrainer.import_client_state]: invalid state type, skip restore.')
            return

        prev_map = state.get('client_prev_params', {})
        fisher_map = state.get('client_fisher_diag', {})

        if not isinstance(prev_map, dict) or not isinstance(fisher_map, dict):
            print('Warning[FedAvgEWCTrainer.import_client_state]: invalid EWC maps, skip restore.')
            return

        restored_prev = {}
        restored_fisher = {}
        for c in self.clients:
            if c.cid in prev_map and c.cid in fisher_map:
                restored_prev[c.cid] = prev_map[c.cid].detach().clone()
                restored_fisher[c.cid] = fisher_map[c.cid].detach().clone()

        self.client_prev_params = restored_prev
        self.client_fisher_diag = restored_fisher

        print(
            f'>>> FedAvg+EWC restored client states: '
            f'{len(self.client_prev_params)} clients with anchors/fisher.'
        )

    def local_train(self, round_i, selected_clients, **kwargs):
        solns = []
        stats = []

        for i, c in enumerate(selected_clients, start=1):
            c.set_flat_model_params(self.latest_model)

            loss_hook = None
            if (
                self.sequential_cl
                and self.lambda_ewc > 0.0
                and c.cid in self.client_prev_params
                and c.cid in self.client_fisher_diag
            ):
                prev_anchor = self.client_prev_params[c.cid].detach().clone()
                fisher_diag = self.client_fisher_diag[c.cid].detach().clone()
                loss_hook = self._build_ewc_loss_hook(prev_anchor, fisher_diag)

            soln, stat = c.local_train(
                loss_hook=loss_hook,
                prev_model=self.worker.options.get('prev_model', None),
                lambda_old=self.worker.options.get('lambda_old', 0.0),
            )

            if self.print_result:
                print("(global)  Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))

            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        self.latest_model = self.worker.get_flat_model_params().detach()

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

        # Sequential task boundary update for next-task EWC.
        self._finalize_ewc_memories()
