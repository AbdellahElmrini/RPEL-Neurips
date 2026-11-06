
import tools
import src
from src import study, misc
tools.success("Module loading...")
import signal, torch
import matplotlib.pyplot as plt

# Set the font family to 'DejaVu Sans'
plt.rcParams['font.family'] = 'DejaVu Sans'


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
dataset = "cifar10"
result_directory = "results_cifar10/results-data-" + dataset + "-neurips-fx"
plot_directory = "results_cifar10/results-plot-" + dataset + "-neurips-fx" 


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



params_cifar = {
  "batch-size": 50, #128, #50
  "loss": "NLLLoss",
  "learning-rate": 0.5,
  "learning-rate-decay": 100 ,#best 750, #50 , 25, 
  "learning-rate-decay-delta" :500 ,# best1500,  # 500, 
  "weight-decay": 1e-2,
  "evaluation-delta": 50,
  "nb-steps": 1000, # Maybe change it back to 1500,
  "momentum-worker": 0.99,
  "numb-labels": 10,
  "mimic-learning-phase": 400,
  "pre-aggregator": "NNM",
  "aggregator": "trmean",
  "method":"cs+",
  "byz-positioning": "target_min_degree"
}



nb_workers = 20
model = "cnn_cifar_old"
#model = "ResNet18"
alphas = [1, 0.5]#10
byzcounts = [3]
nb_neighbors_list = [6]
attacks= [ "dissensus"]#, "ALIE"]#, "SF", "FOE"]
nb_local_steps = [3]#,3]
params = params_common = params_cifar

method = params_cifar["method"]
# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "fx_train_p2p.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=[0,1])
seeds = jobs.get_seeds()

for method in ["RTC", "cs_he", "gts", "cs+"]:#[ "gts", "cs_he", "RTC"]:
  for alpha in alphas:
    for nb_local in nb_local_steps:
      for f in byzcounts:
        for nb_neighbors in nb_neighbors_list:
          for attack in attacks:
            params = params_common.copy()
            params["dataset"] = dataset
            params["model"] = model
            params["nb-workers"] = nb_workers
            params["dirichlet-alpha"] = alpha
            params["nb-decl-byz"] = params["nb-real-byz"] = f
            params["nb-neighbors"] = nb_neighbors
            params["attack"] = attack
            params["b-hat"] = f
            params["nb-local-steps"] = nb_local
            params["method"] = method
            jobs.submit(f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{params['nb-neighbors']}-f_{f}-alpha_{alpha}-b_hat_{params['b-hat']}_nb_local{nb_local}_lr_{params['learning-rate']}-{method}_byz-pos_{params['byz-positioning']}", make_command(params))

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
        
        name = f"{dataset}-n_{params['nb-workers']}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-b_hat_{f}_nb_local{nb_local}_lr_{params['learning-rate']}-{method}_byz-pos_{params['byz-positioning']}"
        brdl = misc.compute_avg_err_op(name, seeds, result_directory, "eval", ("Accuracy", "max"))
        plot.include(brdl[0], "Accuracy", errs="-err", lalp=0.8)
      legend = [f"(attack = {attack})" for attack in attacks]

      plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params['nb-steps'], ymin=0.1, ymax=1.0, legend=legend)
      plot.save(plot_directory + "/" + dataset + "_f=" + str(f) + "_model=" + str(model) + "_neighbors=" + str(nb_neighbors) +"_alpha="+str(alpha)+ ".pdf", xsize=3, ysize=1.5)
