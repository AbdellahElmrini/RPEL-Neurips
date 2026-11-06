# coding: utf-8
###
 # @file train_p2p.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
###

from src import misc, dataset
import tools
tools.success("Module loading...")
import torch, argparse, signal, sys, pathlib, random
from src.byz_attacks import ByzantineAttack
from src.worker_p2p import P2PWorker
from src.byzWorker import ByzantineWorker
import time
import numpy as np
import os

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
	""" Parse the command-line and perform checks.
	Returns:
		Parsed configuration
	"""
	# Description
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--seed",
		type=int,
		default=-1,
		help="Fixed seed to use for reproducibility purpose, negative for random seed")
	parser.add_argument("--device",
		type=str,
		default="auto",
		help="Device on which to run the experiment, \"auto\" by default")
	parser.add_argument("--nb-steps",
		type=int,
		default=1000,
		help="Number of (additional) training steps to do, negative for no limit")
	parser.add_argument("--nb-workers",
		type=int,
		default=15,
		help="Total number of worker machines")
	parser.add_argument("--nb-decl-byz",
		type=int,
		default=0,
		help="Number of Byzantine worker(s) to support")
	parser.add_argument("--nb-real-byz",
		type=int,
		default=0,
		help="Number of actual Byzantine worker(s)")
	parser.add_argument("--aggregator",
		type=str,
		default="average",
		help="(Byzantine-resilient) aggregation rule to use")
	parser.add_argument("--pre-aggregator",
		type=str,
		default=None,
		help="Second (Byzantine-resilient) aggregation rule to use on top of bucketing or NNM")
	parser.add_argument("--bucket-size",
		type=int,
		default=1,
		help="Size of buckets (i.e., number of gradients to average per bucket) in case of bucketing technique")
	parser.add_argument("--attack",
		type=str,
		default=None,
		help="Attack to use")
	parser.add_argument("--model",
		type=str,
		default="simples-full",
		help="Model to train")
	parser.add_argument("--loss",
		type=str,
		default="NLLLoss",
		help="Loss to use")
	parser.add_argument("--dataset",
		type=str,
		default="mnist",
		help="Dataset to use")
	parser.add_argument("--batch-size",
		type=int,
		default=25,
		help="Batch-size to use for training")
	parser.add_argument("--batch-size-test",
		type=int,
		default=100,
		help="Batch-size to use for testing")
	parser.add_argument("--learning-rate",
		type=float,
		default=0.5,
		help="Learning rate to use for training")
	parser.add_argument("--learning-rate-decay",
		type=int,
		default=5000,
		help="Learning rate hyperbolic half-decay time, non-positive for no decay")
	parser.add_argument("--learning-rate-decay-delta",
		type=int,
		default=1,
		help="How many steps between two learning rate updates, must be a positive integer")
	parser.add_argument("--momentum-worker",
		type=float,
		default=0.99,
		help="Momentum on workers to use for training")
	parser.add_argument("--weight-decay",
		type=float,
		default=0,
		help="Weight decay (L2-regularization) to use for training")
	parser.add_argument("--gradient-clip",
		type=float,
		default=None,
		help="Maximum L2-norm, above which clipping occurs, for the estimated gradients")
	#JS: gradient clipping at server pre aggregation
	parser.add_argument("--server-clip",
		action="store_true",
		default=False,
		help="Pre-aggregation robustification layer at the server by gradient clipping")
	parser.add_argument("--result-directory",
		type=str,
		default=None,
		help="Path of the directory in which to save the experiment results (loss, cross-accuracy, ...) and checkpoints, empty for no saving")
	parser.add_argument("--evaluation-delta",
		type=int,
		default=50,
		help="How many training steps between model evaluations, 0 for no evaluation")
	#JS: argument for the heuristic of the mimic attack (duration of learning period)
	parser.add_argument("--mimic-learning-phase",
		type=int,
		default=None,
		help="Number of steps in the learning phase of the mimic heuristic attack")
    #JS: argument for heterogeneous setting
	parser.add_argument("--hetero",
		action="store_true",
		default=False,
		help="Heterogeneous setting")
    #JS: argument for number of labels of dataset (useful for heterogeneity + labelflipping)
	parser.add_argument("--numb-labels",
		type=int,
		default=None,
		help="Number of labels of dataset")
    #JS: argument for distinct datasets for honest workers
	parser.add_argument("--distinct-data",
		action="store_true",
		default=False,
		help="Distinct datasets for honest workers (e.g., privacy setting)")
    #JS: argument for sampling honest data using Dirichlet distribution
	parser.add_argument("--dirichlet-alpha",
		type=float,
		default=None,
		help="The alpha parameter for distribution the data among honest workers using Dirichlet")
    #JS: argument for number of datapoints per honest worker, in case of distinct datasets
	parser.add_argument("--nb-datapoints",
		type=int,
		default=None,
		help="Number of datapoints per honest worker in case of distinct datasets setting (e.g., privacy setting)")
	parser.add_argument("--rag",
		action="store_true",
		default=False,
		help="True if using a robust aggregation rule, otherwise use CG+")
	parser.add_argument("--nb-neighbors", 
		type=int, default=1, 
		help="Number of neighbors to communicate with")
	parser.add_argument("--b-hat",
		type=int, default=0, 
		help="Number of selected Byzantine workers")
	parser.add_argument("--nb-local-steps",
					 type=int, default=1,
		help="Number of local steps to perform before aggregation")
	parser.add_argument("--asynchronous",
		action="store_true",
		default=False,
		help="Use asynchronous DEC with RAG")
	return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
	args = process_commandline()
	# Count the number of real honest workers
	args.nb_honests = args.nb_workers - args.nb_real_byz
	if args.nb_honests < 0:
		tools.fatal(f"Invalid arguments: there are more Byzantine workers ({args.nb_real_byz}) than total workers ({args.nb_workers})")

	cmdline_config = "Configuration" + misc.print_conf((
		("Reproducibility", "not enforced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
		("#workers", args.nb_workers),
		("#neighbors", args.nb_neighbors),
		("#declared Byz.", args.nb_decl_byz),
		("#actually Byz.", args.nb_real_byz),
		("Model", args.model),
		("asynchronous DEC", "yes" if args.asynchronous else "no"),
		("Dataset", (
			("Name", args.dataset),
			("Batch size", (
				("Training", args.batch_size),
				("Testing", args.batch_size_test))))),
		("Loss", (
			("Name", args.loss),
			("L2-regularization", "none" if args.weight_decay is None else f"{args.weight_decay}"))),
		("Optimizer", (
			("Name", "sgd"),
			("Learning rate", args.learning_rate),
			("Momentum", f"{args.momentum_worker}"))),
		("Gradient clip", "no" if args.gradient_clip is None else f"{args.gradient_clip}"),
		("Number of local steps", args.nb_local_steps),
		("Attack", args.attack),
		("Aggregation", args.aggregator),
		("Second Aggregation", args.pre_aggregator),
        ("Extreme Heterogeneity", "yes" if args.hetero else "no"),
        ("Distinct datasets for honest workers",  "yes" if args.distinct_data else "no"),
        ("Dirichlet distribution", "alpha = " + str(args.dirichlet_alpha) if args.dirichlet_alpha is not None else "no")))
	print(cmdline_config)

# ---------------------------------------------------------------------------- #


import json
from collections import Counter
def store_client_label_counts(train_loader_dict, numb_labels=None, output_path=None):
    """Count labels per client and optionally save them as JSON.
    Args:
        train_loader_dict: dict[client_id -> DataLoader]
        numb_labels: total number of classes (optional)
        output_path: file path to save JSON (optional)
    Returns:
        Dict[int, List[int]] where each list is class-counts for that client.
    """
    client_counters = {}
    global_max_label = -1

    for client_id, loader in train_loader_dict.items():
        cnt = Counter()
        for _, y in loader:
            if isinstance(y, torch.Tensor):
                labels = y.view(-1).detach().cpu().tolist()
            else:
                labels = list(y)
            cnt.update(int(v) for v in labels)

        client_counters[client_id] = cnt
        if len(cnt) > 0:
            global_max_label = max(global_max_label, max(cnt.keys()))

    n_classes = numb_labels if numb_labels is not None else (global_max_label + 1)
    label_counts = {}
    for client_id, cnt in client_counters.items():
        label_counts[client_id] = [cnt.get(c, 0) for c in range(n_classes)]

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in label_counts.items()}, f, indent=2)

    return label_counts


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# ...existing code...
def plot_client_label_distribution(json_path, output_path=None):
    """Plot client label distribution as a grid with circles."""
    with open(json_path, "r", encoding="utf-8") as f:
        label_counts = json.load(f)

    label_counts = {int(k): v for k, v in label_counts.items()}
    label_counts = dict(sorted(label_counts.items()))

    num_clients = len(label_counts)
    num_labels = len(next(iter(label_counts.values())))

    fig, ax = plt.subplots(figsize=(max(10, num_clients * 0.8), max(8, num_labels * 0.6)))


    # Use only non-zero counts for normalization (since zeros are hidden)
    all_counts = [c for counts in label_counts.values() for c in counts if c > 0]
    if all_counts:
        min_count = min(all_counts)
        max_count = max(all_counts)
    else:
        min_count, max_count = 1, 1
    denom = (max_count - min_count) if max_count > min_count else 1.0

    cmap = plt.cm.viridis

    # Build arrays for scatter
    xs, ys, sizes, colors = [], [], [], []
    s_min, s_max = 40, 900  # marker area in points^2

    for client_id, counts in label_counts.items():
        for label_id, count in enumerate(counts):
            if count <= 0:
                continue  # hide zero-count circles
            norm = (count - min_count) / denom
            size = s_min + norm * (s_max - s_min)

            xs.append(client_id)
            ys.append(label_id)
            sizes.append(size)
            colors.append(count)

    sc = ax.scatter(
        xs, ys,
        s=sizes,
        c=colors,
        cmap=cmap,
        vmin=min_count,
        vmax=max_count,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
        zorder=3
    )

    ax.set_xlim(-0.5, num_clients - 0.5)
    ax.set_ylim(-0.5, num_labels - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax.set_xlabel('Honest Workers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class Labels', fontsize=12, fontweight='bold')
    ax.set_title('Per-Client Label Distribution', fontsize=14, fontweight='bold')

    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f'C{i}' for i in range(num_clients)])
    ax.set_yticks(range(num_labels))
    ax.set_yticklabels([f'L{i}' for i in range(num_labels)])

    ax.grid(True, which='both', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Number of Samples', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    plt.show()
# ...existing code...
# Setup
tools.success("Experiment setup...")


with tools.Context("setup", "info"):
	# Enforce reproducibility if asked (see https://pytorch.org/docs/stable/notes/randomness.html)
	reproducible = args.seed >= 0
	if reproducible:
		torch.manual_seed(args.seed)
		random.seed(args.seed)
		import numpy
		numpy.random.seed(args.seed)
	torch.backends.cudnn.deterministic = reproducible
	torch.backends.cudnn.benchmark   = not reproducible

	# JS: Create train (one for every honest worker) and test data loaders
	#print("DEBUGGING")
	#print(args.dataset,args.hetero, args.numb_labels, args.dirichlet_alpha, args.distinct_data,args.nb_datapoints,args.nb_honests, args.batch_size, args.batch_size_test)
	train_loader_dict, test_loader = dataset.make_train_test_datasets(args.dataset, heterogeneity=args.hetero,
						numb_labels=args.numb_labels, alpha_dirichlet=args.dirichlet_alpha, distinct_datasets=args.distinct_data,
						nb_datapoints=args.nb_datapoints, honest_workers=args.nb_honests, train_batch=args.batch_size, test_batch=args.batch_size_test)
	

	
	# Make the result directory (if requested)
	if args.result_directory is not None:
		try:
			resdir = pathlib.Path(args.result_directory).resolve()
			resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
			args.result_directory = resdir
		except Exception as err:
			tools.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")

	# Store per-client label distribution
	label_counts_path = (args.result_directory / "client_label_counts.json") if args.result_directory is not None else None
	client_label_counts = store_client_label_counts(train_loader_dict, numb_labels=args.numb_labels, output_path=label_counts_path)
	print("Per-client label counts:")
	for cid in sorted(client_label_counts):
		print(f"Client {cid}: {client_label_counts[cid]}")	

	# Store per-client label distribution
	label_counts_path = (args.result_directory / "client_label_counts.json") if args.result_directory is not None else None
	client_label_counts = store_client_label_counts(train_loader_dict, numb_labels=args.numb_labels, output_path=label_counts_path)
	print("Per-client label counts:")
	for cid in sorted(client_label_counts):
		print(f"Client {cid}: {client_label_counts[cid]}")

	# Plot the distribution
	if label_counts_path is not None:
		plot_output = args.result_directory / "client_label_distribution.png"
		plot_client_label_distribution(str(label_counts_path), output_path=str(plot_output))
	# ---------------------------------------------------------------------------- #
# Training
tools.success("Training...")

start = time.time()
# Training until limit or stopped
with tools.Context("training", "info"):
	fd_eval = (args.result_directory / "eval").open("w") if args.result_directory is not None else None
	fd_eval_worst = (args.result_directory / "eval_worst").open("w") if args.result_directory is not None else None
	if fd_eval is not None:
		misc.make_result_file(fd_eval, ["Step number", "Cross-accuracy"])
		misc.make_result_file(fd_eval_worst , ["Step number", "Cross-accuracy"])

	labelflipping = True if args.attack == "LF" else False

	#JS: Initialize workers, and make them agree on initial parameters
	Workers = list()
	data_samples_per_worker = []  # List to store number of data samples for each honest worker
	for worker_id in range(args.nb_honests):
		#JS: Instantiate worker i
		worker_i = P2PWorker(worker_id, train_loader_dict[worker_id], test_loader, args.nb_workers, args.nb_decl_byz, args.nb_real_byz, args.aggregator, args.pre_aggregator,
					args.server_clip, args.bucket_size, args.model, args.learning_rate, args.learning_rate_decay, args.learning_rate_decay_delta, args.weight_decay,
					args.loss, args.momentum_worker, args.device, labelflipping, args.gradient_clip, args.numb_labels, args.nb_neighbors, args.rag, args.b_hat, args.nb_local_steps, args.asynchronous)
		data_samples_per_worker.append(len(train_loader_dict[worker_id].dataset))  # Track number of samples
		if worker_id > 0:
			#JS: Set model/parameters of worker i to those of worker 0
			worker_i.model.load_state_dict(Workers[0].model.state_dict())
		Workers.append(worker_i)

	#JS: Instantiate Byzantine worker
	byzWorker = ByzantineWorker(args.nb_workers, args.nb_decl_byz, args.nb_real_byz, args.attack, args.aggregator, args.pre_aggregator, args.server_clip,
			     args.bucket_size, Workers[0].model_size, args.mimic_learning_phase, args.gradient_clip, args.device)

	current_step = 0
	#Keeping track of the accuracies of all workers to check the worst one
	accuracies = []
	while not exit_is_requested() and current_step <= args.nb_steps:
		# Evaluate the model if milestone is reached
		milestone_evaluation = args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0		
		if milestone_evaluation:
			accuracy = Workers[0].compute_accuracy()
			print(f"Accuracy (step {current_step})... {accuracy * 100.:.2f}%.")
			accuracies.append([worker.compute_accuracy() for worker in Workers])
			avg_accuracy = sum(accuracies[-1])/len(Workers)
			#avg_accuracy = sum([worker.compute_accuracy() for worker in Workers]) / len(Workers)
			print(f"Average accuracy (step {current_step})... {avg_accuracy * 100.:.2f}%.")
			# Store the evaluation result
			if fd_eval is not None:
				#misc.store_result(fd_eval, current_step, accuracy)
				misc.store_result(fd_eval, current_step, avg_accuracy)
			
			# Compute the consensus error (i.e the average distance between the honest workers' models) and print it

			#avg_model = torch.mean(torch.stack([misc.flatten(worker.model.parameters()) for worker in Workers]), dim=0)
			#consensus_error = sum([torch.norm(misc.flatten(worker.model.parameters()) - avg_model).item() for worker in Workers]) / len(Workers)
			#print(f"Consensus error (step {current_step})... {consensus_error:.4f}.")


        #JS: honest workers perform local step
		honest_local_params = [worker.perform_local_step(current_step) for worker in Workers]
		#honest_local_params = [worker.multiple_local_steps() for worker in Workers]
		
		
		# Update the model on each honest worker
		for worker in Workers:
			worker.aggregate_and_update_parameters(honest_local_params, args, current_step, data_samples_per_worker) 
		# Increase the step counter
		current_step += 1
	for worker in Workers:
		print("Worker", worker.worker_id, "has maximum ", max(worker.num_selected_byz), "selected Byzantines")
	# Storing the evaluation of the worst worker (according to the final accuracy)

	if fd_eval_worst is not None and len(accuracies) > 0:
		print("Storing the evaluation of the worst worker (according to the final accuracy)")
		worst_client = min(range(len(Workers)), key=lambda i: accuracies[-1][i])
		for i, accs in enumerate(accuracies):
			misc.store_result(fd_eval_worst, i * args.evaluation_delta, accs[worst_client])	

	# Storing all the accuracies
	np.save(os.path.join(args.result_directory, "accuracies.npy"), np.array(accuracies)) if args.result_directory is not None else None
tools.success("Finished...")
tools.success(f"Total time: {time.time() - start:.2f} seconds")