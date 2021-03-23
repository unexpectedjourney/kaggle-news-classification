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
