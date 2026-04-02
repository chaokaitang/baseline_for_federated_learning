from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD


class DittoTrainer(BaseTrainer):
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(DittoTrainer, self).__init__(options, dataset, model, self.optimizer)

        # personalization regularizer
        self.lambda_p = options.get('lambda_p', 0.1)

        # initialize personal models for each client (flat tensors)
        init_flat = self.worker.get_flat_model_params().detach()
        self.personal_models = {c.cid: init_flat.clone() for c in self.clients}

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

            # Personalization: update personalized model for selected clients
            for c in selected_clients:
                # Load current personal model into worker
                personal = self.personal_models.get(c.cid)
                if personal is None:
                    personal = self.latest_model.clone()
                # Set client's personalized model into the shared worker for local update
                c.set_flat_model_params(personal)

                # Run local training with proximal term towards current global model
                # Using worker.local_train directly (returns flat solution and stats)
                try:
                    # Run local training for personalization (prox to global)
                    local_solution, _ = self.worker.local_train(c.train_dataloader,
                                                                prox_mu=self.lambda_p,
                                                                global_params=self.latest_model)
                    # Save updated personal model (flat tensor)
                    self.personal_models[c.cid] = local_solution.detach()

                    # (defer evaluation of personalized models until after all personal updates)
                    pass
                except Exception as e:
                    # If personalization fails for any client, skip and keep previous personal model
                    print(f"Warning[DittoTrainer.train]: round={round_i}, cid={c.cid}, personalization failed due to {type(e).__name__}: {e}")

            # After personalization updates, evaluate personalized models for ALL clients
            personal_losses = []
            personal_accs = []
            for c_all in self.clients:
                try:
                    # load personalized model if exists, otherwise use latest global
                    p = self.personal_models.get(c_all.cid, self.latest_model)
                    c_all.set_flat_model_params(p)
                    tot_correct, num_sample, loss = c_all.local_test(use_eval_data=True)
                    acc = tot_correct / num_sample if num_sample > 0 else 0.0
                    avg_loss = float(loss) / float(num_sample) if num_sample > 0 else 0.0
                    personal_losses.append(avg_loss)
                    personal_accs.append(float(acc))
                    # record per-client personalized stat
                    self.metrics.update_personalized_eval_stats(round_i, c_all.cid, avg_loss, float(acc))
                    # restore client's model to global for next operations
                    c_all.set_flat_model_params(self.latest_model)
                except Exception as e:
                    # if evaluation fails for a client, skip it
                    print(f"Warning[DittoTrainer.train]: round={round_i}, cid={c_all.cid}, personalized eval skipped due to {type(e).__name__}: {e}")
                    continue

            # record aggregated personalized statistics (mean and std)
            self.metrics.update_personalized_aggregate(round_i, personal_losses, personal_accs)

            # Learning rate schedule step if any
            try:
                self.optimizer.inverse_prop_decay_learning_rate(round_i)
            except Exception as e:
                print(f"Warning[DittoTrainer.train]: round={round_i}, lr decay failed due to {type(e).__name__}: {e}")

        # Final evaluation
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()
