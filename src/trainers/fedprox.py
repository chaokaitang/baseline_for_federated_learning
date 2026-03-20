from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD

class FedProxTrainer(BaseTrainer):
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # store mu for FedProx proximal term
        self.mu = options.get('mu', 0.0)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedProxTrainer, self).__init__(options, dataset, model, self.optimizer)

    # Further implementation of FedProx specific methods would go here

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Implementation of the training loop specific to FedProx would go here
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        for round_i in range(self.num_round):
            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally (pass proximal term parameters)
            solns, stats = self.local_train(round_i, selected_clients,
                                            prox_mu=self.mu,
                                            global_params=self.latest_model)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train/eval data and persist metrics
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)
        self.metrics.write()


        

