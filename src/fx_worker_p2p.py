import torch, random
#from src.robust_aggregators import RobustAggregator
from src.robust_aggregator import RobustAggregator
from src.byzWorker import ByzantineWorker, DecByzantineWorker, OptimalDissensus
from src.byz_attacks import ByzantineAttack
from src.robust_summations import cs_plus, gts, cs_he, RTC
from . import models, misc
import networkx as nx
import numpy as np
from utils.gossip import LaplacianGossipMatrix


methods_dict={"cs+":cs_plus, "cs_he":cs_he, "gts":gts}




class P2PWorker(object):
    """A worker for decentralized learning in peer to peer model."""

    def __init__(self, worker_id, data_loader, data_loader_test, nb_workers, nb_byz, nb_real_byz, aggregator, pre_aggregator, server_clip, bucket_size, model,
                 learning_rate, learning_rate_decay, learning_rate_decay_delta, weight_decay, loss, momentum, device, labelflipping,
                 gradient_clip, numb_labels, nb_neighbors, rag, b_hat, nb_local_steps, method, comm_graph, dissensus):
        self.worker_id = worker_id
        self.nb_byz = nb_byz
        self.nb_real_byz = nb_real_byz # Number of Byzantine workers that are actually Byzantine
        self.nb_honest = nb_workers - nb_byz 
        self.rag = rag
        self.b_hat = b_hat
        self.comm_graph = comm_graph
        self.method = method
        self.dissensus = dissensus
        #spectrum = nx.laplacian_spectrum(self.comm_graph).astype(np.float32)
        #largest_eig = spectrum[-1]
        #self.rho = (1/largest_eig).astype(np.float32)
        self.rho = 1.0
        self.W = LaplacianGossipMatrix(self.comm_graph)
        self.W = torch.tensor(self.W).to(device)

        self.loaders = {"train": data_loader, "test": data_loader_test}
        self.iterators = {"train": iter(data_loader), "test": iter(data_loader_test)}

        self.initial_learning_rate = self.current_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delta = learning_rate_decay_delta

        self.device = device
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)()
        self.model.to(self.device)
        #JS: list of shapes of the model in question. Used when unflattening gradients and model parameters
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.model_size = len(misc.flatten(self.model.parameters()))

        if self.device == "cuda":
            #JS: model is on GPU and not explicitly restricted to one particular card => enable data parallelism
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])
        #self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay, momentum=momentum)

        #self.robust_aggregator = RobustAggregator(aggregator, pre_aggregator, server_clip, nb_neighbors+1-b_hat, b_hat, bucket_size, self.model_size, self.device)
        
        #################
        #aggregator_info = {"name": str(aggregator), "parameters": {"f": self.b_hat}}
        #pre_aggregator_list = [{"name": str(pre_aggregator),  "parameters": {"f": self.b_hat}}] if pre_aggregator is not None else []
        #pre_aggregator_list = []
        #self.robust_aggregator = RobustAggregator(aggregator_info, pre_aggregator_list)#, pre_aggregator, server_clip, nb_neighbors+1-b_hat, b_hat, bucket_size, self.model_size, self.device)

        #################
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = momentum
        self.gradient_clip = gradient_clip

        self.labelflipping = labelflipping
        self.numb_labels = numb_labels

        self.nb_neighbors =  len(list((self.comm_graph.neighbors(self.worker_id))))
        self.nb_local_steps = nb_local_steps
        self.num_selected_byz = []
        metropolis = True
        if metropolis:
            self.byz_weights = 0
            neighbors = list(self.comm_graph.neighbors(self.worker_id))
            neighbors_degrees = sorted([comm_graph.degree(i) for i in neighbors])
            pivot_degree = comm_graph.degree(self.worker_id)
            for i in range(b_hat):
                try:
                    self.byz_weights += 1/(max(neighbors_degrees[i], pivot_degree) +1)
                except:
                    #raise ValueError(f"b_hat = {b_hat} is too large compared to the number of neighbors of worker {self.worker_id}, which is {len(neighbors)}")
                    print(f"Warning: b_hat = {b_hat} is too large compared to the number of neighbors of worker {self.worker_id}, which is {len(neighbors)}")
                    self.byz_weights += 0
        else:
            self.byz_weights = b_hat
        
        self.num_clipped = []

    # Sample train or test batch, depending on the mode
    # mode can be "train" or "test"
    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])


    #Generic function to compute gradient on batch = (inputs, targets)
    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()])


    #JCompute honest gradient and flipped gradient (if required)
    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        
        #compute gradient on flipped labels
        if self.labelflipping:
            # in case batch norm is used, set the model in eval mode when computing the flipped gradient,
            # in order not to change the running mean and variance
            self.model.eval()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            self.gradient_labelflipping = self.backward_pass(inputs, targets_flipped)
            self.model.train()

        #JS: compute honest gradient (i.e., on current parameters and on true labels)
        return self.backward_pass(inputs, targets)


    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1-self.momentum)

        if self.gradient_clip is not None:
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient


    def local_model_update(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            else:
                return self.current_learning_rate

        #Update the learning rate
        new_learning_rate = update_learning_rate(current_step)
        if self.current_learning_rate != new_learning_rate:
            self.current_learning_rate = new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_learning_rate

        # Perform the update step
        self.optimizer.step()


    def perform_local_step(self, current_step):
        for _ in range(self.nb_local_steps):
            self.set_gradient(self.compute_momentum())
            self.local_model_update(current_step)
        return misc.flatten(self.model.parameters())


    def multiple_local_steps(self):
        for _ in range(self.nb_local_steps):
            self.model.train()
            inputs, targets = self.sample_batch("train")
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
        return misc.flatten(self.model.parameters())

 
    def aggregate_and_update_parameters(self, honest_params, args, current_step):
        #if self.method =="RTC":
        #    self.aggregate_and_update_parameters_RTC(honest_params, args, current_step)
        #else:
        if self.dissensus:
            self.aggregate_and_update_parameters_fixed_graph_dissensus(honest_params, args, current_step)
        elif args.attack == "opt_dissensus":
            self.aggregate_and_update_parameters_fixed_graph_opt_dissensus(honest_params, args, current_step)
        else:
            self.aggregate_and_update_parameters_fixed_graph(honest_params, args, current_step)
    
    @torch.no_grad()
    def aggregate_and_update_parameters_fixed_graph_opt_dissensus(self, honest_params, args, current_step):

        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))

        neighbors.append(self.worker_id)

        honest_neighbors= [ i for i in neighbors if i < self.nb_honest]
        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)

        self.num_selected_byz.append(nb_selected_byz)

        #with torch.no_grad():
        byzWorker = OptimalDissensus(target = self.worker_id, honest_neighbors = honest_neighbors, nb_honest = self.nb_honest, nb_byz_neighbors = nb_selected_byz, pivot_params=pivot_params, network = self.comm_graph, method= self.method
                                        )
    

        
        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]

        byzantine_params = byzWorker(honest_params)

        #worker_params = honest_local_params + [byz_param for byz_param in byzantine_params] 
        # Build mapping of neighbor index to byzantine param index
        byz_idx = 0
        worker_params = []
        for neighbor in neighbors:
            if neighbor < self.nb_honest:
                worker_params.append(honest_params[neighbor])
            else:
                worker_params.append(byzantine_params[byz_idx])
                byz_idx += 1

        with torch.no_grad():
            if self.method =="RTC":
                aggregation_ = RTC()
                aggregate_params = aggregation_(gradients= worker_params, local_gradient = pivot_params, weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
                                         byzantine_weight_budget= self.byz_weights, communication_step=self.rho)
                self.num_clipped.append(aggregation_.num_removed)
                self.set_model_parameters(aggregate_params)
            elif self.method in methods_dict:
                worker_params = torch.stack(worker_params)
                differences = worker_params - pivot_params
                
                robust_aggregate, num_clipped = methods_dict[self.method](weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),#.to(self.device), 
                                        gradients=differences, 
                                        byz_weights=self.byz_weights) 
                self.num_clipped.append(num_clipped)
                aggregate_params= pivot_params + self.rho *robust_aggregate
                self.set_model_parameters(aggregate_params)
            else:
                raise ValueError(f"Unknown aggregation method {self.method}")

        
    def aggregate_and_update_parameters_RTC(self, honest_params, args, current_step):

        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))

        neighbors.append(self.worker_id)

        #honest_neighbors= [ i for i in neighbors if i < self.nb_honest]
        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)

        self.num_selected_byz.append(nb_selected_byz)


        byzWorker = ByzantineWorker(len(neighbors), nb_selected_byz, nb_selected_byz, args.attack, args.aggregator, args.pre_aggregator, 
                  model_size = self.model_size, mimic_learning_phase= args.mimic_learning_phase, gradient_clip= args.gradient_clip, device=args.device,
                  server_clip= args.server_clip, bucket_size = args.bucket_size)
        
        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]

        byzantine_params = byzWorker.compute_byzantine_vectors(honest_local_params, None, current_step)

        #worker_params = honest_local_params + [byz_param for byz_param in byzantine_params] 
        # Build mapping of neighbor index to byzantine param index
        byz_idx = 0
        worker_params = []
        for neighbor in neighbors:
            if neighbor < self.nb_honest:
                worker_params.append(honest_params[neighbor])
            else:
                worker_params.append(byzantine_params[byz_idx])
                byz_idx += 1


        with torch.no_grad():
            #worker_params = torch.stack(worker_params)
            aggregation_ = RTC()
            aggregate_params = aggregation_(gradients= worker_params, local_gradient = pivot_params, weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
                                     byzantine_weight_budget= self.byz_weights, communication_step=self.rho)
            #print(aggregate_params)
            #aggregate_params, num_clipped = aggregation_(worker_params,weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
            #                         honest_index = self.worker_id, byzantine_weight_budget= self.byz_weights, communication_step=self.rho)
            self.num_clipped.append(aggregation_.num_removed) 
            self.set_model_parameters(aggregate_params)
        



    def aggregate_and_update_parameters_fixed_graph(self, honest_params, args, current_step):

        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))

        neighbors.append(self.worker_id)

        #honest_neighbors= [ i for i in neighbors if i < self.nb_honest]
        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)

        self.num_selected_byz.append(nb_selected_byz)
        

        byzWorker = ByzantineWorker(len(neighbors), nb_selected_byz, nb_selected_byz, args.attack, args.aggregator, args.pre_aggregator, 
			      model_size = self.model_size, mimic_learning_phase= args.mimic_learning_phase, gradient_clip= args.gradient_clip, device=args.device,
                  server_clip= args.server_clip, bucket_size = args.bucket_size)
        
        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]

        byzantine_params = byzWorker.compute_byzantine_vectors(honest_local_params, None, current_step)

        # Build mapping of neighbor index to byzantine param index
        byz_idx = 0
        worker_params = []
        for neighbor in neighbors:
            if neighbor < self.nb_honest:
                worker_params.append(honest_params[neighbor])
            else:
                worker_params.append(byzantine_params[byz_idx])
                byz_idx += 1


        with torch.no_grad():
            if self.method =="RTC":
                aggregation_ = RTC()
                aggregate_params = aggregation_(gradients= worker_params, local_gradient = pivot_params, weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
                                         byzantine_weight_budget= self.byz_weights, communication_step=self.rho)
                self.num_clipped.append(aggregation_.num_removed) 
                self.set_model_parameters(aggregate_params)
            elif self.method in methods_dict:
                worker_params = torch.stack(worker_params)
                differences = worker_params - pivot_params
            
                robust_aggregate, num_clipped = methods_dict[self.method](weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),#.to(self.device), 
                                    gradients=differences, 
                                    byz_weights=self.byz_weights) 
                self.num_clipped.append(num_clipped)
                aggregate_params= pivot_params + self.rho *robust_aggregate
                self.set_model_parameters(aggregate_params)
            else:
                raise ValueError(f"Unknown aggregation method {self.method}")
    
    @torch.no_grad()
    def aggregate_and_update_parameters_fixed_graph_dissensus(self, honest_params, args, current_step):

        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))

        neighbors.append(self.worker_id)

        honest_neighbors= [ i for i in neighbors if i < self.nb_honest]
        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)

        self.num_selected_byz.append(nb_selected_byz)

        #with torch.no_grad():
        byzWorker = DecByzantineWorker(target = self.worker_id, honest_neighbors = honest_neighbors, nb_honest = self.nb_honest, nb_byz_neighbors = nb_selected_byz, pivot_params=pivot_params, network = self.comm_graph,
                                        device= self.device, epsilon=1)
    

        
        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]

        byzantine_params = byzWorker.compute_byzantine_vectors(honest_local_params)

        #worker_params = honest_local_params + [byz_param for byz_param in byzantine_params] 
        # Build mapping of neighbor index to byzantine param index
        byz_idx = 0
        worker_params = []
        for neighbor in neighbors:
            if neighbor < self.nb_honest:
                worker_params.append(honest_params[neighbor])
            else:
                worker_params.append(byzantine_params[byz_idx])
                byz_idx += 1

        with torch.no_grad():
            if self.method =="RTC":
                aggregation_ = RTC()
                aggregate_params = aggregation_(gradients= worker_params, local_gradient = pivot_params, weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
                                         byzantine_weight_budget= self.byz_weights, communication_step=self.rho)
                self.num_clipped.append(aggregation_.num_removed)
                self.set_model_parameters(aggregate_params)
            elif self.method in methods_dict:
                worker_params = torch.stack(worker_params)
                differences = worker_params - pivot_params
                
                robust_aggregate, num_clipped = methods_dict[self.method](weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),#.to(self.device), 
                                        gradients=differences, 
                                        byz_weights=self.byz_weights) 
                self.num_clipped.append(num_clipped)
                aggregate_params= pivot_params + self.rho *robust_aggregate
                self.set_model_parameters(aggregate_params)
            else:
                raise ValueError(f"Unknown aggregation method {self.method}")


    @torch.no_grad()
    def compute_accuracy(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in self.loaders["test"]:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total


    def set_gradient(self, gradient):
        """ Overwrite the gradient with the given one."""
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()


    def set_model_parameters(self, params):
        params = misc.unflatten(params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = params[j].data.detach().clone()