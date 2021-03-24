import datetime
from pathlib import Path

from cv_preparation import cv_evaluation_preparation
from nlp_preparation import nlp_evaluation_preparation
from src.eval import summarize_cv_predictions


def all_evaluation_preparation(
        cv_batch_size,
        cv_model_name,
        cv_model_path,
        cv_coef,
        nlp_batch_size,
        nlp_model_name,
        nlp_max_seq_len,
        nlp_model_path,
        nlp_coef,
        n_classes,
        device
):
    submissions = Path("submissions") / "combined"
    model_name = "combined"
    _, cv_probs = cv_evaluation_preparation(
        cv_batch_size,
        cv_model_name,
        cv_model_path,
        n_classes,
        device
    )

    cv_probs[[0, 1, 3, 4, 5, 6]] *= cv_coef

    _, nlp_probs = nlp_evaluation_preparation(
        nlp_batch_size,
        nlp_model_name,
        nlp_model_path,
        nlp_max_seq_len,
        n_classes,
        device
    )
    nlp_probs[[0, 1, 3, 4, 5, 6]] *= nlp_coef
    final_df = cv_probs.groupby('Id').sum().add(nlp_probs.groupby('Id').sum(), fill_value=0).reset_index()
    # final_df = cv_probs.append(nlp_probs)
    final_df = summarize_cv_predictions(final_df)
    final_df = final_df[["Id", "Predicted"]]
    final_df.to_csv(
        submissions / f"{model_name}-{int(datetime.datetime.now().timestamp())}.csv",
        index=False)
