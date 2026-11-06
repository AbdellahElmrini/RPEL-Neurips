
import tools
import src
from src import study, misc
tools.success("Module loading...")
import signal, torch


import matplotlib.pyplot as plt

# Set the font family to 'DejaVu Sans'
plt.rcParams['font.family'] = 'DejaVu Sans'
import os

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #


#Pick the dataset on which to run experiments
dataset = "mnist"
result_directory = "results_mnist/results-data-" + dataset + "-neurips-seeds"
plot_directory = "results_mnist/results-plot-" + dataset + "-neurips-seeds"



with tools.Context("cmdline", "info"):
  args = misc.process_commandline()
  # Make the result directories
  args.result_directory = misc.check_make_dir(result_directory)
  args.plot_directory = misc.check_make_dir(plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# Base parameters for the MNIST experiments
params_mnist = {
  "batch-size": 25,
  "loss": "NLLLoss",
  "weight-decay": 1e-4,
  "nb-steps": 100, # 200 for small number of workers
  "momentum-worker": 0.9,
  "numb-labels": 10,
  "pre-aggregator": "NNM",
  "aggregator" : "standard", 
  "evaluation-delta":10,
  "byz-positioning": "random"
}


nb_workers = 30 
model = "cnn_mnist"
alpha = 1

byzcounts = [1]#,10] #[20] #
b_hat_list = [1]#[9] #

nb_neighbors_list = [3]#[20] 
nb_local_steps = [1]

attacks= [ "ALIE"]#, "SF", "FOE"] "dissensus",
#method = params_mnist["method"]

methods =  ["RTC", "cs_he", "gts", "cs+"]
params_common = params_mnist

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "fx_train_p2p.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=[0,1,2,3,4])#,1])
seeds = jobs.get_seeds()


#for i, f in enumerate(byzcounts):
for method in methods:
  for nb_local in nb_local_steps:
    for i,f in enumerate(byzcounts):
      for nb_neighbors in nb_neighbors_list:
        #nb_neighbors = nb_neighbors_list[i]
        for attack in attacks:
          params = params_common.copy()
          params["dataset"] = dataset
          params["model"] = model
          params["nb-workers"] = nb_workers
          params["dirichlet-alpha"] = alpha
          params["nb-decl-byz"] = params["nb-real-byz"] = f
          params["nb-neighbors"] = nb_neighbors
          params["attack"] = attack
          params["b-hat"] = b_hat_list[i]
          params["nb-local-steps"] = nb_local
          params["method"] = method
          jobs.submit(f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{params['nb-neighbors']}-f_{f}-alpha_{alpha}-nb-local_{nb_local}-{method}-byz-pos_{params['byz-positioning']}", make_command(params)) 

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")


for nb_local in nb_local_steps:
  for f in byzcounts:
    for nb_neighbors in nb_neighbors_list:
      plot = study.LinePlot()
      for attack in attacks:
        
        name = f"{dataset}-n_{params['nb-workers']}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}-{method}-byz-pos_{params['byz-positioning']}"
        brdl = misc.compute_avg_err_op(name, seeds, result_directory, "eval", ("Accuracy", "max"))
        plot.include(brdl[0], "Accuracy", errs="-err", lalp=0.8)
        #legend = [f"(f = {f})"]
      legend = [f"(attack = {attack})" for attack in attacks]
      plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params['nb-steps'], ymin=0.1, ymax=1.0, legend=legend)
      plot.save(plot_directory + "/" + dataset + "_n=" + str(nb_workers) + "_f=" + str(f) + "_model=" + str(model) + "_neighbors=" + str(nb_neighbors) +"_method=" +str(method) + ".pdf", xsize=3, ysize=1.5)
