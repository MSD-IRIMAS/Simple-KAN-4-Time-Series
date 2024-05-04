import pandas as pd
import numpy as np
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from models import KAN_CLASSIFIER

from utils.utils import load_data, create_directory

from aeon.transformations.collection.feature_based import Catch22

from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression


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

        _accs = []
        _accs_lr = []

        for _run in range(args.runs):
            kan_classifier = KAN_CLASSIFIER(
                width=args.width,
                output_dir=output_dir_dataset,
                steps=args.steps,
                k=args.k,
                grid=args.grid,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
            )

            _, accuracy_test = kan_classifier.fit(
                xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest
            )

            _accs.append(accuracy_test)

            catch22_transformer = Catch22(
                use_pycatch22=True,
            )
            xtrain_transformed = catch22_transformer.fit_transform(
                np.expand_dims(xtrain, axis=1)
            )
            xtest_transformed = catch22_transformer.transform(
                np.expand_dims(xtest, axis=1)
            )

            lr_classifier = LogisticRegression(
                penalty='none',
                multi_class="multinomial",
                random_state=rng.randint(0, np.iinfo(np.int32).max),
            )
            lr_classifier.fit(X=xtrain_transformed, y=ytrain)
            ypred = lr_classifier.predict(X=xtest_transformed)

            _accs_lr.append(accuracy_score(y_pred=ypred, y_true=ytest, normalize=True))

        df = pd.DataFrame(
            columns=[
                "accuracy-mean",
                "accuracy-std",
                "accuracy-lr-mean",
                "accuracy-lr-std",
            ]
        )
        df.loc[len(df)] = {
            "accuracy-mean": np.mean(_accs),
            "accuracy-std": np.std(_accs),
            "accuracy-lr-mean": np.mean(_accs_lr),
            "accuracy-lr-std": np.std(_accs_lr),
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
