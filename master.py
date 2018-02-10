from run_experiment import Experiment

exp = Experiment("experiments/es-en/data/es.train",
                 "experiments/es-en/data/en.train",
                 "experiments/es-en/data/es.dev",
                 "experiments/es-en/data/en.dev",
                 #dest_lang = "en",
                 joint_codes = "experiments/es-en/data/es-en.yml",
                 model_prefix = "experiments/es-en/models/es-en",
                 train_log_prefix = "experiments/es-en/logs/es-en",
                 translation_dir = "experiments/es-en/trans"
                )

exp.run_experiment()
