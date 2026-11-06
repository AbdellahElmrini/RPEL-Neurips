import torch
from . import models, misc


class Worker(object):
    """A worker for distributed learning in parameter server model."""
    def __init__(self, data_loader, test_loader, batch_size, model, model_size, loss, momentum, labelflipping,
                 gradient_clip, numb_labels, privacy_multiplier, bit_precision, gradient_clamp, device, batch_norm):

        self.data_loader = data_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.number_batches = len(data_loader)
        self.train_iterator = iter(data_loader)

        self.device = device
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)()
        self.model.to(self.device)
        self.model_size = model_size
        #JS: dictionary to hold the running mean and variance of the model, in case batch norm is used
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_mean = dict()
            self.batch_var = dict()

        self.momentum = momentum
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.gradient_clip = gradient_clip

        self.labelflipping = labelflipping
        self.numb_labels = numb_labels

        #JS: security parameters (homomorphic encryption)
        self.bit_precision, self.gradient_clamp = bit_precision, gradient_clamp
        if self.bit_precision is not None and self.gradient_clamp is not None:
            self.quantization = (2**(self.bit_precision - 1) - 1) / self.gradient_clamp

        #JS: privacy parameters
        self.privacy_std = None
        if privacy_multiplier is not None:
            from opacus.accountants.rdp import RDPAccountant
            self.privacyAccountant = RDPAccountant()
            self.privacy_multiplier = privacy_multiplier
            self.sample_rate = 1 / self.number_batches
            self.privacy_std = 2 * self.gradient_clip * privacy_multiplier / self.batch_size
            self.grad_noise = torch.distributions.normal.Normal(torch.zeros(self.model_size, device=self.device),
                                                                torch.ones(self.model_size, device=self.device).mul_(self.privacy_std))


    def sample_train_batch(self):
        try:
            return next(self.train_iterator)
        except:
            self.train_iterator = iter(self.data_loader)
            return next(self.train_iterator)


    #JS: Generic function to compute gradient on batch = (inputs, targets)
    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()])


    def compute_private_gradient(self, inputs, targets):
        self.privacyAccountant.step(noise_multiplier=self.privacy_multiplier, sample_rate=self.sample_rate)

        avg_gradient = torch.zeros(self.model_size, device=self.device)
        for input, target in zip(inputs, targets):
            #JS: compute the per sample gradient on (input, target) and add it to avg_gradient
            target_tensor = torch.tensor([target.item()], device=self.device)
            avg_gradient.add_(misc.clip_vector(self.backward_pass(input, target_tensor), self.gradient_clip))

        #JS: compute the average of all per sample gradients
        avg_gradient.div_(self.batch_size)
        #JS: inject noise to the average gradient
        avg_gradient.add_(self.grad_noise.sample())
        return avg_gradient


    #JS: Compute honest (potentially private) gradient and flipped gradient (if required)
    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        #JS: compute gradient on flipped labels
        if self.labelflipping:
            #JS: in case batch norm is used, set the model in eval mode when computing the flipped gradient,
            # in order not to change the running mean and variance
            self.model.eval()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            self.gradient_LF = self.backward_pass(inputs, targets_flipped)
            self.model.train()

        #JS: compute honest (and potentially private) gradient (i.e., on current parameters and on true labels)
        if self.privacy_std is not None:
            return self.compute_private_gradient(inputs, targets)
        return self.backward_pass(inputs, targets)


    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1-self.momentum)

        if self.gradient_clamp is not None:
            #JS: Need to clamp the momentum_gradient (and quantize it in case of encryption)
            return self.quantize_gradient(self.momentum_gradient)

        elif self.gradient_clip is not None and self.privacy_std is None:
            #JS: only clip the gradient when no privacy is used (because otherwise only the per sample gradients must be clipped)
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient


    #JS: Clamp the gradient (and flipped gradient if LF is enabled) and quantize it in case of encryption
    def quantize_gradient(self, gradient):
        clamped_grad = torch.clamp(gradient, min=-self.gradient_clamp, max=self.gradient_clamp)
        if self.labelflipping:
            self.gradient_LF = torch.clamp(self.gradient_LF, min=-self.gradient_clamp, max=self.gradient_clamp)

        if self.bit_precision is not None:
            torch.round_(clamped_grad.mul_(self.quantization))
            if self.labelflipping:
                torch.round_(self.gradient_LF.mul_(self.quantization))
        return clamped_grad


    def get_privacy_budget(self, delta=1e-4):
        epsilon, _ = self.privacyAccountant.get_privacy_spent(delta=delta)
        return epsilon


    def set_model_parameters(self, server_params):
        """ Overwrite the model parameters with the ones sent by the server."""
        for param, server_param in zip(self.model.parameters(), server_params):
            param.data = server_param.data.detach().clone()


    #JS: Store the running mean and variance of every batch norm layer in a dictionary, before sending them to the server
    def compute_running_mean_var(self):
        for param_tensor in self.model.state_dict():
            if "running_mean" in param_tensor:
                self.batch_mean[param_tensor] = self.model.state_dict()[param_tensor]
            elif "running_var" in param_tensor:
                self.batch_var[param_tensor] = self.model.state_dict()[param_tensor]

    @torch.no_grad()
    def compute_accuracy(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total