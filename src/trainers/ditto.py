from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD


class DittoTrainer(BaseTrainer):
    """Ditto trainer with persistent personalized models.

    Notes:
    - Global branch follows standard FL local training + aggregation.
    - Personalized branch performs proximal local optimization around current
      global model using each client's persistent personal model as init.
    - `lambda_old` and other CL-related options are framework-level extensions.
      Set `lambda_old=0` to recover plain Ditto-style personalized objective.
    """

    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(DittoTrainer, self).__init__(options, dataset, model, self.optimizer)

        # personalization regularizer
        self.lambda_p = options.get('lambda_p', 0.1)
        # Independent local budget for personalized branch.
        # Fallback is aligned in main.read_options, keep this guard for robustness.
        self.personal_num_epoch = int(options.get('personal_num_epoch', options.get('num_epoch', 1)))
        if self.personal_num_epoch <= 0:
            print(
                f"Warning[DittoTrainer.__init__]: invalid personal_num_epoch={self.personal_num_epoch}, "
                f"fallback to num_epoch={options.get('num_epoch', 1)}"
            )
            self.personal_num_epoch = int(options.get('num_epoch', 1))

        # initialize personal models for each client (flat tensors)
        init_flat = self.worker.get_flat_model_params().detach()
        self.personal_models = {c.cid: init_flat.clone() for c in self.clients}

    def export_client_state(self):
        """Export per-client personal states for sequential CL task transfer."""
        return {
            'personal_models': {cid: v.detach().clone() for cid, v in self.personal_models.items()}
        }

    def import_client_state(self, state):
        """Restore per-client personal states for sequential CL task transfer."""
        if not isinstance(state, dict):
            print("Warning[DittoTrainer.import_client_state]: invalid state type, skip restore.")
            return

        src = state.get('personal_models', {})
        if not isinstance(src, dict):
            print("Warning[DittoTrainer.import_client_state]: invalid `personal_models`, skip restore.")
            return

        restored = {}
        for c in self.clients:
            if c.cid in src:
                restored[c.cid] = src[c.cid].detach().clone()
            else:
                restored[c.cid] = self.latest_model.detach().clone()
        self.personal_models = restored

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):
            # Test latest model on train and eval data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose clients
            selected_clients = self.select_clients(seed=round_i)

            # Global training: each client performs local update (standard local objective)
            solns, stats = self.local_train(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)

            # Aggregate global model
            self.latest_model = self.aggregate(solns)
            print('=' * 102 + "\n")

            # Personalization: update personalized model for selected clients
            # Ditto-specific: use an independent local budget (`personal_num_epoch`)
            # for personalized proximal optimization.
            global_anchor = self.latest_model.detach().clone()
            for i, c in enumerate(selected_clients, start=1):
                # Load current personal model into worker
                personal = self.personal_models.get(c.cid)
                if personal is None:
                    personal = global_anchor.clone()
                # Set client's personalized model into the shared worker for local update
                c.set_flat_model_params(personal)

                # Run local training with proximal term towards current global model
                # Using worker.local_train directly (returns flat solution and stats)
                old_num_epoch = self.worker.num_epoch
                try:
                    self.worker.num_epoch = int(self.personal_num_epoch)
                    # Run local training for personalization (prox to current global anchor)
                    soln_personal, local_stats = c.local_train(
                        prox_mu=self.lambda_p,
                        global_params=global_anchor,
                        prev_model=self.worker.options.get('prev_model', None),
                        lambda_old=self.worker.options.get('lambda_old', 0.0),
                    )
                    _, local_solution = soln_personal
                    if self.print_result:
                        print("(Private) Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                            "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                            "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                            round_i, c.cid, i, self.clients_per_round,
                            local_stats['norm'], local_stats['min'], local_stats['max'],
                            local_stats['loss'], local_stats['acc']*100, local_stats['time']))
                    # Save updated personal model (flat tensor)
                    self.personal_models[c.cid] = local_solution.detach().clone()

                    # (defer evaluation of personalized models until after all personal updates)
                    pass
                except Exception as e:
                    # If personalization fails for any client, skip and keep previous personal model
                    print(f"Warning[DittoTrainer.train]: round={round_i}, cid={c.cid}, personalization failed due to {type(e).__name__}: {e}")
                finally:
                    self.worker.num_epoch = old_num_epoch

            # After personalization updates, evaluate personalized models for ALL clients
            self._evaluate_personalized_all_clients(round_i)

            # Learning rate schedule step if any
            try:
                self.optimizer.inverse_prop_decay_learning_rate(round_i)
            except Exception as e:
                print(f"Warning[DittoTrainer.train]: round={round_i}, lr decay failed due to {type(e).__name__}: {e}")

        # Final evaluation
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)
        # Ensure personalized metrics at final index (num_round) are populated too.
        self._evaluate_personalized_all_clients(self.num_round)

        # Save tracked information
        self.metrics.write()

    def _evaluate_personalized_all_clients(self, round_i):
        personal_losses = []
        personal_accs = []
        for c_all in self.clients:
            try:
                # Load personalized model if exists, otherwise fallback to latest global.
                p = self.personal_models.get(c_all.cid, self.latest_model)
                c_all.set_flat_model_params(p)
                tot_correct, num_sample, loss = c_all.local_test(use_eval_data=True)
                acc = tot_correct / num_sample if num_sample > 0 else 0.0
                avg_loss = float(loss) / float(num_sample) if num_sample > 0 else 0.0
                personal_losses.append(avg_loss)
                personal_accs.append(float(acc))
                self.metrics.update_personalized_eval_stats(round_i, c_all.cid, avg_loss, float(acc))
                c_all.set_flat_model_params(self.latest_model)
            except Exception as e:
                print(
                    f"Warning[DittoTrainer._evaluate_personalized_all_clients]: "
                    f"round={round_i}, cid={c_all.cid}, personalized eval skipped due to {type(e).__name__}: {e}"
                )
                continue

        self.metrics.update_personalized_aggregate(round_i, personal_losses, personal_accs)
