from experiment import Experiment

exp = Experiment("experiments/es-en/data/es.train.tok.tc",
                 "experiments/es-en/data/en.train.tok.tc",
                 "experiments/es-en/data/es.dev.sgm",
                 "experiments/es-en/data/en.dev.sgm",
                 "experiments/es-en/data/es.dev.tok.tc",
                 #dest_lang = "en",
                 joint_codes = "experiments/es-en/data/es-en.codes",
                 model_prefix = "experiments/es-en/models/es-en",
                 train_log_prefix = "experiments/es-en/logs/es-en",
                 vocab_dir = "experiments/es-en/vocab",
                 translation_dir = "experiments/es-en/trans"
                )

exp.run_experiment()
