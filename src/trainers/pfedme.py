from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch


class PFedMeTrainer(BaseTrainer):
    """A minimal pFedMe implementation compatible with this baseline.

    Behavior (approximate, suitable for baseline comparisons):
    - Each client k maintains a personalized model v_k (flat tensor).
    - Local training solves: min_w f_k(w) + (lambda_p/2) * ||w - v_k||^2 by passing prox_mu and global_params=v_k to worker.local_train.
    - Server aggregates client local solutions (w_k) into global model w (same as FedAvg).
    - Personal models v_k are updated by a relaxed step towards the client's local solution:
        v_k <- v_k - beta * (v_k - w_k)
    - After each round, evaluate global model and personalized models on clients' own test sets and record metrics.
    """

    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.print_result = not options['noprint']

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(PFedMeTrainer, self).__init__(options, dataset, model, self.optimizer)

        # pFedMe hyperparameters
        self.lambda_p = options.get('lambda_p', 0.1)
        # CLI --eta is the personal-model inner-step size (eta). server mixing uses server_beta.
        # For backward compatibility, accept 'eta' or 'beta' in options.
        self.eta = options.get('eta', options.get('beta', 0.5))
        self.server_beta = options.get('server_beta', 1.0)

        # initialize personalized models for each client (flat tensors)
        init_flat = self.worker.get_flat_model_params().detach()
        self.personal_models = {c.cid: init_flat.clone() for c in self.clients}

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):
            # Test latest model on train data and eval data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose clients
            selected_clients = self.select_clients(seed=round_i)

            # For global aggregation, collect local solutions and stats
            solns, stats = [], []
            for i, c in enumerate(selected_clients, start=1):
                # send global model to client worker
                c.set_flat_model_params(self.latest_model)

                # run local training with proximal term towards personal model v_k
                v_k = self.personal_models.get(c.cid, self.latest_model)
                try:
                    soln, stat = c.local_train(prox_mu=self.lambda_p, global_params=v_k)
                except TypeError as e:
                    # older Client interface may expect different signature; try worker call
                    print(f"Warning[PFedMeTrainer.train]: round={round_i}, cid={c.cid}, fallback to worker.local_train due to {type(e).__name__}: {e}")
                    soln, stat = self.worker.local_train(c.train_dataloader, prox_mu=self.lambda_p, global_params=v_k)

                if self.print_result:
                    print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                        "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                        "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                        round_i, c.cid, i, self.clients_per_round,
                        stat['norm'], stat['min'], stat['max'],
                        stat['loss'], stat['acc']*100, stat['time']))
                solns.append(soln)
                stats.append(stat)
                num_sample, local_solution = soln

                # update personal model v_k towards local solution following pFedMe:
                # v_k <- v_k - eta * lambda_p * (v_k - theta_tilde)
                # here `self.eta` is the inner-loop step size and `self.lambda_p` is lambda
                try:
                    updated_vk = v_k - (self.eta * self.lambda_p) * (v_k - local_solution.detach())
                except Exception as e:
                    print(f"Warning[PFedMeTrainer.train]: round={round_i}, cid={c.cid}, personal model update fallback due to {type(e).__name__}: {e}")
                    updated_vk = local_solution.detach()
                self.personal_models[c.cid] = updated_vk

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Aggregate global model following pFedMe-style mixing:
            # Aggregate global model using the trainer-specific aggregate method
            self.latest_model = self.aggregate(solns)

            # After aggregation evaluate personalized models for all clients and record
            personal_losses = []
            personal_accs = []
            for c_all in self.clients:
                try:
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
                    print(f"Warning[PFedMeTrainer.train]: round={round_i}, cid={c_all.cid}, personalized eval skipped due to {type(e).__name__}: {e}")
                    continue

            self.metrics.update_personalized_aggregate(round_i, personal_losses, personal_accs)

            # Learning rate schedule step if any
            try:
                self.optimizer.inverse_prop_decay_learning_rate(round_i)
            except Exception as e:
                print(f"Warning[PFedMeTrainer.train]: round={round_i}, lr decay failed due to {type(e).__name__}: {e}")

        # Final evaluation
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns):
        """Aggregate local solutions into new global model following pFedMe mixing rule.

        Args:
            solns: list of tuples (num_sample, flat_tensor) where flat_tensor is a torch tensor

        Returns:
            new_flat_global: torch tensor for new global parameters
        """
        # If no solns, return existing global
        if solns is None or len(solns) == 0:
            return self.latest_model

        # Build aggregated client solution (either simple average or weighted by samples)
        agg = None
        if self.simple_average:
            # simple average over clients
            for _, solution in solns:
                if agg is None:
                    agg = solution.clone()
                else:
                    agg = agg + solution
            agg = agg / float(len(solns))
        else:
            # weighted by number of samples
            total = 0.0
            for num, solution in solns:
                total += float(num)
                if agg is None:
                    agg = (float(num) * solution.clone())
                else:
                    agg = agg + float(num) * solution
            if total > 0:
                agg = agg / float(total)
            else:
                agg = agg / float(len(solns))

        # Ensure agg is on same device as latest_model
        try:
            agg = agg.to(self.latest_model.device)
        except Exception as e:
            print(f"Warning[PFedMeTrainer.aggregate]: failed to align device due to {type(e).__name__}: {e}")

        # Mix previous global model and aggregated client solution
        try:
            new_global = (1.0 - self.server_beta) * self.latest_model + self.server_beta * agg
        except Exception as e:
            print(f"Warning[PFedMeTrainer.aggregate]: global mixing fallback due to {type(e).__name__}: {e}")
            new_global = agg.detach()

        return new_global.detach()
