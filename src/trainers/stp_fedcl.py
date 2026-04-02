from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch
from torch.utils.data import DataLoader, Subset


class STPFedCLTrainer(BaseTrainer):
    """Spatio-Temporal Personalized Federated Continual Learning trainer.

    Global branch:
        standard FL update to produce aggregated global model.

    Personalized branch for each selected client k:
        L = L_task
          + mu * ||p_k - w_g||^2
          + lambda_s * ||p_k - p_k_prev||^2
          + lambda_l * ||p_k - p_k_ema||^2
        NOTE: all L2 penalties here are unnormalized squared L2 penalties,
              implemented with torch.sum((...) ** 2).

    Beta selection (inference-only):
        y_hat = (1-beta_k) * f(x; w_g) + beta_k * f(x; p_k)
    """

    def __init__(self, options, dataset):
        self.options = options
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(STPFedCLTrainer, self).__init__(options, dataset, model, self.optimizer)

        # Spatial/temporal regularization coefficients
        self.mu = float(options.get('mu', 0.0))
        self.lambda_s = float(options.get('lambda_s', 0.0))
        self.lambda_l = float(options.get('lambda_l', 0.0))
        self.alpha = float(options.get('alpha', 0.9))

        # Beta selection policy
        self.beta_mode = str(options.get('beta_mode', 'adaptive_search')).lower()
        self.beta_fixed = float(max(0.0, min(1.0, float(options.get('beta_fixed', 0.5)))))
        beta_candidates = options.get('beta_candidates', [0.0, 0.25, 0.5, 0.75, 1.0])
        self.beta_candidates = [float(max(0.0, min(1.0, float(v)))) for v in beta_candidates]
        if len(self.beta_candidates) == 0:
            self.beta_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.beta_val_ratio = float(options.get('beta_val_ratio', 0.1))
        self.log_reg_terms = bool(options.get('log_reg_terms', False))
        self.reg_log_every = max(1, int(options.get('reg_log_every', 20)))

        # Client states
        init_flat = self.worker.get_flat_model_params().detach()
        self.personal_models = {c.cid: init_flat.clone() for c in self.clients}
        self.client_prev = {c.cid: init_flat.clone() for c in self.clients}
        self.client_ema = {c.cid: init_flat.clone() for c in self.clients}
        self.client_betas = {c.cid: float(self.beta_fixed) for c in self.clients}

    @staticmethod
    def _flat_params_with_grad(model):
        return torch.cat([p.view(-1) for p in model.parameters()])

    def _build_personal_loss_hook(self, global_model, prev_anchor, ema_anchor, round_i=None, cid=None):
        step_counter = {'n': 0}

        def _hook(worker, base_loss):
            current = self._flat_params_with_grad(worker.model)
            reg = torch.tensor(0.0, device=current.device, dtype=current.dtype)

            reg_mu = torch.tensor(0.0, device=current.device, dtype=current.dtype)
            reg_s = torch.tensor(0.0, device=current.device, dtype=current.dtype)
            reg_l = torch.tensor(0.0, device=current.device, dtype=current.dtype)

            if self.mu > 0.0:
                g = global_model
                if g.device != current.device:
                    g = g.to(current.device)
                reg_mu = 0.5 * self.mu * torch.sum((current - g) ** 2)
                reg = reg + reg_mu

            if self.lambda_s > 0.0:
                p = prev_anchor
                if p.device != current.device:
                    p = p.to(current.device)
                reg_s = 0.5 * self.lambda_s * torch.sum((current - p) ** 2)
                reg = reg + reg_s

            if self.lambda_l > 0.0:
                e = ema_anchor
                if e.device != current.device:
                    e = e.to(current.device)
                reg_l = 0.5 * self.lambda_l * torch.sum((current - e) ** 2)
                reg = reg + reg_l

            if self.log_reg_terms:
                step_counter['n'] += 1
                if step_counter['n'] % self.reg_log_every == 0:
                    try:
                        print(
                            f"[RegTerms] round={round_i} cid={cid} step={step_counter['n']} "
                            f"base={float(base_loss.item()):.6f} "
                            f"mu={float(reg_mu.item()):.6f} "
                            f"short={float(reg_s.item()):.6f} "
                            f"long={float(reg_l.item()):.6f} "
                            f"total_reg={float(reg.item()):.6f}"
                        )
                    except Exception:
                        pass

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

    def _make_client_val_loader(self, client):
        # IMPORTANT: beta search must not use test data.
        # Use a deterministic validation split from client TRAIN data.
        ds = client.train_data
        n = len(ds)
        if n == 0:
            return None

        ratio = min(max(self.beta_val_ratio, 0.0), 0.9)
        val_n = max(1, int(round(n * ratio)))
        val_n = min(val_n, n)

        # deterministic client-wise split for reproducibility
        gen = torch.Generator()
        gen.manual_seed(int(self.options['seed']) + int(client.cid) + 97)
        perm = torch.randperm(n, generator=gen).tolist()
        val_idx = perm[:val_n]

        val_ds = Subset(ds, val_idx)
        return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

    def export_client_state(self):
        """Export minimal per-client state for sequential CL task transfer."""
        return {
            'personal_models': {cid: v.detach().clone() for cid, v in self.personal_models.items()},
            'client_prev': {cid: v.detach().clone() for cid, v in self.client_prev.items()},
            'client_ema': {cid: v.detach().clone() for cid, v in self.client_ema.items()},
            'client_betas': {cid: float(v) for cid, v in self.client_betas.items()},
        }

    def import_client_state(self, state):
        """Restore per-client state for sequential CL task transfer."""
        if not isinstance(state, dict):
            print("Warning[STPFedCLTrainer.import_client_state]: invalid state type, skip restore.")
            return

        def _restore_tensor_map(src_map, dst_map_name):
            src = state.get(src_map, {})
            if not isinstance(src, dict):
                print(f"Warning[STPFedCLTrainer.import_client_state]: invalid `{src_map}`, skip.")
                return {}
            restored = {}
            for c in self.clients:
                if c.cid in src:
                    restored[c.cid] = src[c.cid].detach().clone()
                else:
                    restored[c.cid] = self.latest_model.detach().clone()
            setattr(self, dst_map_name, restored)

        _restore_tensor_map('personal_models', 'personal_models')
        _restore_tensor_map('client_prev', 'client_prev')
        _restore_tensor_map('client_ema', 'client_ema')

        betas = state.get('client_betas', {})
        if not isinstance(betas, dict):
            print("Warning[STPFedCLTrainer.import_client_state]: invalid `client_betas`, use default.")
            betas = {}
        self.client_betas = {c.cid: float(betas.get(c.cid, self.beta_fixed)) for c in self.clients}

    def _eval_mixed_logits(self, dataloader, global_flat, personal_flat, beta):
        if dataloader is None:
            return 0.0, float('inf'), 0

        total = 0
        total_correct = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in dataloader:
                x = self.worker.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.worker.set_flat_model_params(global_flat)
                pred_global = self.worker.model(x)
                self.worker.set_flat_model_params(personal_flat)
                pred_personal = self.worker.model(x)

                pred_mix = (1.0 - beta) * pred_global + beta * pred_personal
                pred_mix, y_local = self.worker._apply_task_aware_logits_labels(pred_mix, y)

                loss = criterion(pred_mix, y_local)
                _, pred_label = torch.max(pred_mix, 1)
                correct = pred_label.eq(y_local).sum().item()

                bs = y_local.size(0)
                total += bs
                total_correct += correct
                total_loss += float(loss.item()) * bs

        if total == 0:
            return 0.0, float('inf'), 0
        return float(total_correct) / float(total), float(total_loss) / float(total), int(total)

    def _select_client_beta(self, client):
        if self.beta_mode == 'fixed':
            return float(self.beta_fixed)

        val_loader = self._make_client_val_loader(client)
        if val_loader is None:
            return float(self.beta_fixed)

        global_flat = self.latest_model.detach().clone()
        personal_flat = self.personal_models.get(client.cid, global_flat).detach().clone()

        best_beta = float(self.beta_fixed)
        best_acc = -1.0
        best_loss = float('inf')

        for beta in self.beta_candidates:
            b = float(beta)
            acc, loss, _ = self._eval_mixed_logits(val_loader, global_flat, personal_flat, b)
            if (acc > best_acc) or (acc == best_acc and loss < best_loss):
                best_acc = acc
                best_loss = loss
                best_beta = b

        return float(best_beta)

    def _update_all_client_betas(self):
        for c in self.clients:
            self.client_betas[c.cid] = self._select_client_beta(c)

        # keep worker params on latest global model for consistency
        self.worker.set_flat_model_params(self.latest_model)

    def _evaluate_personalized_all_clients(self, round_i):
        losses = []
        accs = []
        for c in self.clients:
            p = self.personal_models.get(c.cid, self.latest_model)
            c.set_flat_model_params(p)
            tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
            acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
            avg_loss = float(loss) / float(num_sample) if num_sample > 0 else 0.0
            losses.append(avg_loss)
            accs.append(float(acc))
            self.metrics.update_personalized_eval_stats(round_i, c.cid, avg_loss, float(acc))

        self.metrics.update_personalized_aggregate(round_i, losses, accs)
        self.worker.set_flat_model_params(self.latest_model)

    def _evaluate_collab_all_clients(self, round_i):
        losses = []
        accs = []
        global_flat = self.latest_model.detach().clone()
        for c in self.clients:
            beta = float(self.client_betas.get(c.cid, self.beta_fixed))
            personal = self.personal_models.get(c.cid, self.latest_model).detach().clone()
            eval_loader = DataLoader(c.test_data, batch_size=self.batch_size, shuffle=False)
            try:
                acc, loss, _ = self._eval_mixed_logits(eval_loader, global_flat, personal, beta)
                self.metrics.update_collab_eval_stats(round_i, c.cid, float(loss), float(acc))
                losses.append(float(loss))
                accs.append(float(acc))
            except Exception as e:
                print(f"Warning[STPFedCLTrainer._evaluate_collab_all_clients]: round={round_i}, cid={c.cid}, error={type(e).__name__}: {e}")

        self.metrics.update_collab_aggregate(round_i, losses, accs)
        self.worker.set_flat_model_params(self.latest_model)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        self.latest_model = self.worker.get_flat_model_params().detach()
        # Conservative starting point under sum-scale L2 penalties:
        # try mu/lambda_s/lambda_l in [1e-6, 1e-4] first.

        for round_i in range(self.num_round):
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            selected_clients = self.select_clients(seed=round_i)

            # Branch A: global update and aggregation
            solns, stats = self.local_train(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)

            # Branch B: personalized update with spatio-temporal regularization
            global_anchor = self.latest_model.detach().clone()
            for c in selected_clients:
                personal = self.personal_models.get(c.cid, global_anchor).detach().clone()
                c.set_flat_model_params(personal)

                prev_anchor = self.client_prev.get(c.cid, personal).detach().clone()
                ema_anchor = self.client_ema.get(c.cid, personal).detach().clone()
                loss_hook = self._build_personal_loss_hook(global_anchor,
                                                           prev_anchor,
                                                           ema_anchor,
                                                           round_i=round_i,
                                                           cid=c.cid)

                local_solution, _ = self.worker.local_train(c.train_dataloader, loss_hook=loss_hook)
                self.personal_models[c.cid] = local_solution.detach().clone()
                self._update_client_memories(c.cid, local_solution)

            # Track personalized metrics each round
            self._evaluate_personalized_all_clients(round_i)
            self._update_all_client_betas()
            self._evaluate_collab_all_clients(round_i)

            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Adaptive beta selection once at training end (or fixed assignment)
        self._update_all_client_betas()

        self.metrics.write()
