import pandas as pd
import numpy as np
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from models import KAN_CLASSIFIER, LP_CLASSIFIER

from utils.utils import load_data, create_directory

from sklearn.utils import check_random_state


@hydra.main(config_name="config_hydra.yaml", config_path="config")
def main(args: DictConfig):
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    xtrain, ytrain, xtest, ytest = load_data(file_name=args.dataset)

    output_dir = args.output_dir
    create_directory(output_dir)

    rng = check_random_state(args.random_state)

    if args.task_to_solve == "classification":
        output_dir_task = output_dir + args.task_to_solve + "/"
        create_directory(output_dir_task)

        output_dir_dataset = output_dir_task + args.dataset + "/"
        create_directory(output_dir_dataset)

        _accs_kan = []
        _accs_lp = []

        for _ in range(args.runs):
            kan_classifier = KAN_CLASSIFIER(
                width=args.width,
                output_dir=output_dir_dataset,
                steps=args.steps,
                k=args.k,
                grid=args.grid,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
            )

            _, accuracy_kan_test = kan_classifier.fit_and_validate(
                xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest
            )

            _accs_kan.append(accuracy_kan_test)

            lp_classifier = LP_CLASSIFIER(
                random_state=rng.randint(0, np.iinfo(np.int32).max)
            )
            _, accuracy_lp_test = lp_classifier.fit_and_validate(
                xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest
            )

            _accs_lp.append(accuracy_lp_test)

        df = pd.DataFrame(
            columns=[
                "accuracy-kan-mean",
                "accuracy-kan-std",
                "accuracy-lp-mean",
                "accuracy-lp-std",
            ]
        )
        df.loc[len(df)] = {
            "accuracy-kan-mean": np.mean(_accs_kan),
            "accuracy-kan-std": np.std(_accs_kan),
            "accuracy-lp-mean": np.mean(_accs_lp),
            "accuracy-lp-std": np.std(_accs_lp),
        }

        df.to_csv(output_dir_dataset + "/results.csv", index=False)

        if args.get_formulas:
            formulas = kan_classifier.get_symbolic_function()

            for i, string in enumerate(formulas):
                filename = os.path.join(output_dir_dataset, f"formula_class_{i+1}.txt")
                string = str(string)
                with open(filename, "w") as file:
                    file.write(string)


if __name__ == "__main__":
    main()
