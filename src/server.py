import torch, random

from src.robust_aggregators import RobustAggregator
from . import models, misc
class Server(object):
    """Parameter server for distributed training."""
    def __init__(self, aggregator, pre_aggregator, server_clip, momentum_server, model, device, nb_workers, nb_byz, bucket_size, subsampling,
                 weight_decay, learning_rate, learning_rate_decay, learning_rate_decay_delta, bit_precision, gradient_clamp, test_loader, model_scaling=None):

        self.current_learning_rate = self.initial_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delta = learning_rate_decay_delta

        self.device = device
        self.model = getattr(models, model)()
        self.model.to(self.device)

        self.model.eval()
        self.test_loader = test_loader
        if self.device == "cuda":
            #JS: model is on GPU and not explicitly restricted to one particular card => enable data parallelism
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])
        self.model_size = len(misc.flatten(self.model.parameters()))
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay)

        if model_scaling is not None:
            #JS: update initial model parameters by multiplicatively scaling them
            initial_model = misc.flatten(self.model.parameters()) * model_scaling
            self.set_model_parameters(initial_model)

        self.nb_workers = nb_workers
        self.nb_byz = nb_byz

        #JS: Instantiate the robust aggregator to be used
        self.robust_aggregator = RobustAggregator(aggregator, pre_aggregator, server_clip, nb_workers, nb_byz, bucket_size, 
                                                  self.model_size, device)
        self.momentum_server = momentum_server
        self.aggregate_momentum = torch.zeros(self.model_size, device=device)

        #JS: boolean that is true when subsampling is enabled on the server, prior to the robust aggregation
        self.subsampling = subsampling

        #JS: security parameters (homomorphic encryption)
        #JS: if bit_precision < 0, then no dequantization is done at the server.
        self.bit_precision = bit_precision
        if bit_precision is not None:
            self.quantization = (2**(self.bit_precision - 1) - 1) / gradient_clamp


    def compute_aggregate_momentum(self, worker_momentums):

        if self.subsampling and 2 * self.nb_byz + 1 < self.nb_workers:
            #JS: subsample (2f+1) workers
            subsampled_workers = random.sample(range(self.nb_workers), 2 * self.nb_byz + 1)
            worker_momentums = [worker_momentums[i] for i in subsampled_workers]

        #JS: aggregate the incoming momentums
        aggregate_gradient = self.robust_aggregator.aggregate(worker_momentums)
        #JS: compute the momentum at the server (if required)
        self.aggregate_momentum.mul_(self.momentum_server)
        self.aggregate_momentum.add_(aggregate_gradient, alpha=1-self.momentum_server)

        #JS: need to dequantize if aggregate_momentum is encrypted
        if self.bit_precision is not None:
            self.aggregate_momentum.mul_(1 / self.quantization)

        # Update the gradient locally
        self.set_gradient(self.aggregate_momentum)


    def compute_aggregate_mean_var(self, honest_worker_means, honest_worker_vars, byzantine_dict_mean, byzantine_dict_var):
        #JS: aggregate running means and vars (in case of batch norm), and update state_dict of model
        updated_state_dict = self.model.state_dict()
        for batch_layer in honest_worker_means[0].keys():
            honest_means = [honest_worker_means[worker][batch_layer] for worker in range(len(honest_worker_means))]
            updated_state_dict[batch_layer] = self.robust_aggregator.aggregate(honest_means + byzantine_dict_mean[batch_layer])
        for batch_layer in honest_worker_vars[0].keys():
            honest_vars = [honest_worker_vars[worker][batch_layer] for worker in range(len(honest_worker_vars))]
            updated_state_dict[batch_layer] =  self.robust_aggregator.aggregate(honest_vars + byzantine_dict_var[batch_layer])
        self.model.load_state_dict(updated_state_dict)


    def set_gradient(self, gradient):
        """ Overwrite the gradient with the given one."""
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()


    def update_parameters(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            else:
                return self.current_learning_rate

        #JS: Update the learning rate
        new_learning_rate = update_learning_rate(current_step)
        if self.current_learning_rate != new_learning_rate:
            self.current_learning_rate = new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_learning_rate

        #regularized_momentum = torch.add(self.aggregate_momentum, self.parameters, alpha=5e-4)
        #self.parameters.add_(regularized_momentum, alpha=-self.current_learning_rate)
        #self.set_model_parameters(self.parameters)

        # Perform the update step
        self.optimizer.step()
    

    def set_model_parameters(self, server_params):
        """ Overwrite the model parameters with the ones sent by the server."""
        server_params = misc.unflatten(server_params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = server_params[j].data.clone().detach()

    @torch.no_grad()
    def compute_accuracy(self):
        total = 0
        correct = 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total