import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_nlp(model, data_loader, device, pbar_desc="eval phase"):
    model.eval()
    predictions = []
    prediction_probs = []

    with torch.no_grad():
        for element in tqdm(data_loader, desc=pbar_desc):
            input_ids = element["input_ids"].to(device)
            attention_mask = element["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return predictions, prediction_probs


def eval_cv(model, data_loader, device, pbar_desc="eval phase"):
    model.eval()

    with torch.no_grad():
        submission_df = pd.DataFrame()

        for (idx, images, _) in tqdm(data_loader, desc=pbar_desc):
            images = images.to(device)
            outputs = model(images)
            # outputs = outputs.cpu().numpy()

            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            probs = probs.cpu().numpy()
            probs = probs.T

            result_dict = {
                key: prob for key, prob in enumerate(probs)
            }
            result_dict["Id"] = idx
            submission_df = submission_df.append(pd.DataFrame(result_dict))

        return submission_df


def summarize_cv_predictions(submission_df):
    submission_df = submission_df.groupby("Id").mean().idxmax(axis="columns")
    submission_df = submission_df.to_frame().reset_index()
    submission_df = submission_df.rename(columns={0: "Predicted"})

    return submission_df
