import torch, random
#from src.robust_aggregators import RobustAggregator
from src.byzWorker import ByzantineWorker
from src.byz_attacks import ByzantineAttack

from src.robust_aggregator import RobustAggregator
from . import models, misc

class P2PWorker(object):
    """A worker for decentralized learning in peer to peer model."""

    def __init__(self, worker_id, data_loader, data_loader_test, nb_workers, nb_byz, nb_real_byz, aggregator, pre_aggregator,server_clip, bucket_size, model,
                 learning_rate, learning_rate_decay, learning_rate_decay_delta, weight_decay, loss, momentum, device, labelflipping,
                 gradient_clip, numb_labels, nb_neighbors, rag, b_hat, nb_local_steps):
        self.worker_id = worker_id
        self.nb_byz = nb_byz
        self.nb_real_byz = nb_real_byz # Number of Byzantine workers that are actually Byzantine
        self.nb_honest = nb_workers - nb_byz 
        self.rag = rag
        self.b_hat = b_hat

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
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay, momentum=momentum)
        #self.criterion = torch.nn.CrossEntropyLoss()
        #JS: Instantiate the robust aggregator to be used
        #self.robust_aggregator = RobustAggregator(aggregator, pre_aggregator, server_clip, self.nb_honest, nb_byz, bucket_size, self.model_size, self.device)
        #self.robust_aggregator = RobustAggregator(aggregator, pre_aggregator, server_clip, nb_neighbors+1-b_hat, b_hat, bucket_size, self.model_size, self.device)
        #################
        aggregator_info = {"name": str(aggregator), "parameters": {"f": self.b_hat}}
        pre_aggregator_list = [{"name": str(pre_aggregator),  "parameters": {"f": self.b_hat}}]
        self.robust_aggregator = RobustAggregator(aggregator_info, pre_aggregator_list)#, pre_aggregator, server_clip, nb_neighbors+1-b_hat, b_hat, bucket_size, self.model_size, self.device)

        #################
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = momentum
        self.gradient_clip = gradient_clip

        self.labelflipping = labelflipping
        self.numb_labels = numb_labels

        self.nb_neighbors = nb_neighbors # TODO : later we can set it to log(nb_workers) or something like that
        self.nb_local_steps = nb_local_steps
        self.num_selected_byz = []
    #JS: Sample train or test batch, depending on the mode
    #JS: mode can be "train" or "test"
    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])


    #JS: Generic function to compute gradient on batch = (inputs, targets)
    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()])


    #JS: Compute honest gradient and flipped gradient (if required)
    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        #JS: compute gradient on flipped labels
        if self.labelflipping:
            #JS: in case batch norm is used, set the model in eval mode when computing the flipped gradient,
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

        #JS: Update the learning rate
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
    
    def one_local_step(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
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
        if self.rag:
            #print("Using RAG")
            self.aggregate_and_update_parameters_dec_with_rag(honest_params, args, current_step)
        else:
            #print("Using CG+")
            self.aggregate_and_update_parameters_cgplus(honest_params, args, current_step)

    def aggregate_and_update_parameters_cgplus(self, honest_params, args, current_step):
        
        # Implementing CG+ algorithm 
        # Randomly select nb_neighbors neighbors to aggregate with (some of them may be Byzantine)
        indices_list = list(range(self.nb_honest + self.nb_byz))
        indices_list.remove(self.worker_id)
        
        random_indices = random.sample(indices_list, self.nb_neighbors)
        selected_byzantine_indices = [i - self.nb_honest for i in random_indices if i >= self.nb_honest]
        nb_selected_byz = len(selected_byzantine_indices)
        byzWorker = ByzantineWorker(len(random_indices), nb_selected_byz, nb_selected_byz, args.attack, args.aggregator, args.pre_aggregator, args.server_clip,
			     args.bucket_size, self.model_size, args.mimic_learning_phase, args.gradient_clip, args.device)
        #print("Number of selected Byzantines", len(selected_byzantine_indices))
        self.num_selected_byz.append(nb_selected_byz)

        
        # Move only needed parameters to GPU (avoid O(N²) memory)
        honest_local_params = [honest_params[i].to(self.device) for i in random_indices if i < self.nb_honest]
        """
        labelflipping = True if args.attack == "LF" else False
        if labelflipping:
			#In case of labelflipping attack, do something particular
            flipped_gradients = [worker.gradient_labelflipping for worker in Workers]
            attack = ByzantineAttack("LF", args.nb_real_byz,self.model_size, None, -1, args.gradient_clip, None)
            byzantine_flipped_gradient = attack.generate_byzantine_vectors(None, flipped_gradients, -1)[0]
            byzantine_param = torch.stack(honest_local_params).mean(dim=0).add(byzantine_flipped_gradient, alpha=-Workers[0].current_learning_rate)
            byzantine_params = [byzantine_param] * args.nb_real_byz
        else:""" # Currently not supporting label flipping
            
        byzantine_params = byzWorker.compute_byzantine_vectors(honest_local_params, None, current_step)
               
        # Build worker_params by reusing honest_local_params + byzantine params
        worker_params = honest_local_params + byzantine_params
        

        worker_params = torch.stack(worker_params)
        differences = worker_params - honest_params[self.worker_id].to(self.device)
        distances = differences.norm(dim=1)
        #clipping_threshold = torch.topk(distances,2*self.nb_byz).values[-1] 
        #clipping_threshold = torch.topk(distances,2*self.b_hat).values[-1] 
        clipping_threshold = torch.topk(distances,2*self.b_hat).values[-1]  if self.b_hat > 0 else torch.inf
        #clipping_threshold = torch.topk(distances,self.b_hat+1).values[-1] 
        mask = distances[:, None].broadcast_to(differences.shape) > clipping_threshold

        clipped_differences = torch.where(
                                        mask,  # Compare each norm to the threshold
                                        differences * (clipping_threshold / distances[:, None]),  # Scale down the vector
                                        differences  # Otherwise, keep it unchanged
                                    )
        # Testing communication learning rate scheduler
        communication_lr = 1/(current_step//250+1)
        aggregate_params = honest_params[self.worker_id].to(self.device) + communication_lr* clipped_differences.sum(dim=0) * (1/(self.nb_neighbors))

        self.set_model_parameters(aggregate_params)

    def aggregate_Vanilla_FL(self, honest_params, args, current_step):
        # Aggregate all incoming parameters
        aggregate_params = self.robust_aggregator.aggregate([honest_params[i] for i in range(len(honest_params))])

        # Update the model parameters
        self.set_model_parameters(aggregate_params)
        self.num_selected_byz.append(0)

    def aggregate_and_update_parameters_dec_with_rag(self, honest_params, args, current_step):
        # Randomly select nb_neighbors neighbors to aggregate with (some of them may be Byzantine)
        # Avoid creating full indices_list, use set difference instead
        all_indices = set(range(self.nb_honest + self.nb_byz))
        all_indices.discard(self.worker_id)
        random_indices = random.sample(list(all_indices), self.nb_neighbors)
        
        # Count Byzantine workers without creating intermediate list
        nb_selected_byz = sum(1 for i in random_indices if i >= self.nb_honest)
        self.num_selected_byz.append(nb_selected_byz)
        
        byzWorker = ByzantineWorker(len(random_indices), nb_selected_byz, nb_selected_byz, args.attack, args.aggregator, args.pre_aggregator, args.server_clip,
			     args.bucket_size, self.model_size, args.mimic_learning_phase, args.gradient_clip, args.device) ###
        
        # Extract honest params and move to GPU only when needed (avoid O(N²) memory)
        honest_local_params = [honest_params[i] for i in random_indices if i < self.nb_honest] ###
        byzantine_params = byzWorker.compute_byzantine_vectors(honest_local_params, None, current_step)
        
        # Build worker_params efficiently: reuse honest_local_params instead of duplicating list comprehension
        worker_params = honest_local_params + byzantine_params
        worker_params.append(honest_params[self.worker_id]) ###

        # Aggregate all incoming parameters
        
        #aggregate_params = self.robust_aggregator.aggregate(worker_params)
        aggregate_params = self.robust_aggregator.aggregate_vectors(worker_params) ########################
        # Update the model parameters and cleanup intermediate tensors
        self.set_model_parameters(aggregate_params.to(self.device))
        del honest_local_params, byzantine_params, worker_params, aggregate_params


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