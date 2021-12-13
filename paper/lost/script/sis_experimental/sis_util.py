"""Util for running SIS on PyTorch models."""

import numpy as np
import torch

from sufficient_input_subsets import sis


def predict(model, inputs, add_softmax=False):
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        if add_softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)
    return preds


def make_f_for_class(model, class_idx, batch_size=128, add_softmax=False):
    def f(inputs):
        with torch.no_grad():
            ret_np = False
            if isinstance(inputs, np.ndarray):
                ret_np = True
                inputs = torch.from_numpy(inputs).cuda()
            else:
                inputs = inputs.cuda()
            num_batches = int(np.ceil(inputs.shape[0] / batch_size))
            all_preds = []
            for batch_idx in range(num_batches):
                batch_start_i = batch_idx * batch_size
                batch_end_i = min(inputs.shape[0],
                                  (batch_idx + 1) * batch_size)
                assert batch_end_i > batch_start_i
                preds = predict(
                    model,
                    inputs[batch_start_i:batch_end_i],
                    add_softmax=add_softmax)[:, class_idx]
                all_preds.append(preds)
            all_preds = torch.cat(all_preds)
            if ret_np:
                all_preds = all_preds.cpu().numpy()
            return all_preds
    return f


def find_sis_on_input(model, x, initial_mask, fully_masked_input, threshold,
                      batch_size=128, add_softmax=False):
    """Find first SIS on input x with PyTorch model."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).cuda()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).cuda())[0]
        pred_class = int(pred.argmax())
        pred_confidence = float(pred.max())
    if pred_confidence < threshold:
        return None
    f_class = make_f_for_class(model, pred_class, batch_size=batch_size,
                               add_softmax=add_softmax)
    sis_result = sis.find_sis(
        f_class,
        threshold,
        x.cpu().numpy(),
        initial_mask,
        fully_masked_input,
    )
    return sis_result


def create_masked_input(x, sis_result, fully_masked_input):
    return sis.produce_masked_inputs(
        x.cpu().numpy(), fully_masked_input, [sis_result.mask])[0]


def save_sis_result(sr, filepath):
    np.savez_compressed(
        filepath,
        sis=np.array(sr.sis),
        ordering_over_entire_backselect=sr.ordering_over_entire_backselect,
        values_over_entire_backselect=sr.values_over_entire_backselect,
        mask=sr.mask,
    )


def load_sis_result(filepath):
    loaded = np.load(filepath)
    sr = sis.SISResult(
        sis=loaded['sis'],
        ordering_over_entire_backselect=(
            loaded['ordering_over_entire_backselect']),
        values_over_entire_backselect=loaded['values_over_entire_backselect'],
        mask=loaded['mask'],
    )
    return sr
