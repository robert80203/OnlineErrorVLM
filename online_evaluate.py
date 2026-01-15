import json
import os
from dataclasses import asdict
import torch.utils.data as data
from transformers.trainer_pt_utils import LabelSmoother
import torch

from data import (
    build_concat_train_dataset,
    build_eval_dataset_dict,
    get_compute_metrics_dict,
    get_data_collator,
)
from engine import (
    StopEvaluationAfterOneStepCallback,
    StopTrainingAfterOneStepCallback,
    # TrainerWithGenToEval,
    TrainerWithOnlineGenToEval,
)

from models import build_model_and_tokenizer, count_parameters, parse_args

def simple_data_collator(batch: list[list]):
    batch = list(zip(*batch))
    batch_conversation, batch_frames, batch_sample_idxs, batch_evaluation_kwargs = batch
    return batch_conversation[0], batch_frames[0], batch_sample_idxs[0], batch_evaluation_kwargs[0]


def data_processing(batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs, tokenizer):
    batch = tokenizer(
        batch_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )

    batch_labels = torch.full_like(
        batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long
    )

    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text,
        batch_labels,
        batch.input_ids,
        batch.offset_mapping,
        batch_learn_ranges,
    ):
        output = []
        for learn_r in learn_range:
            # Find the minimum index in offset_mapping[:, 0] that is >= learn_r.start
            start_candidates = torch.nonzero(
                offset_mapping[:, 0] >= learn_r.start, as_tuple=False
            )
            start = (
                start_candidates.min().item() if start_candidates.numel() > 0 else 0
            )

            # Find the minimum closest stop index
            if offset_mapping[:, 0][-1] >= learn_r.stop:
                stop_candidates = torch.nonzero(
                    offset_mapping[:, 1] <= learn_r.stop, as_tuple=False
                )
                stop = (
                    stop_candidates.max().item()
                    if stop_candidates.numel() > 0
                    else len(input_ids)
                )
                if start == stop:
                    stop = stop + 1 if stop < len(input_ids) else start - 1
            else:  # the last eos token
                stop = len(input_ids)
            # print(labels.size(), input_ids.size())
            # print(start, stop)
            labels[start - 1 : stop - 1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id

            output.append([start, stop, input_ids[start:stop]])
            if stop == len(input_ids):
                output[-1].append(labels)

    batch["labels"] = batch_labels
    batch.pop("offset_mapping")

    batch["frames"] = torch.cat(batch_frames)


    batch["bboxes"] = None

    batch["sample_idxs"] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        if "stream" in batch_evaluation_kwargs[0]["evaluator"]:
            batch_evaluation_kwargs[0]["load_ranges"] = batch_load_ranges[0]
        batch["evaluation_kwargs"] = batch_evaluation_kwargs[
            0
        ]  # evaluation only supports bs = 1, so its okay
    return batch

def train():
    args = parse_args()
    args.run_name = args.output_dir.split("/")[-1]
    args.logging_dir = os.path.join(args.output_dir, "runs")
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    # _ = count_parameters(model, layers=False)

    train_dataset = build_concat_train_dataset(
        tokenizer=tokenizer,
        transform=(
            model.vision_processor if hasattr(model, "vision_processor") else None
        ),
        **asdict(args),
    )
    eval_dataset_dict = build_eval_dataset_dict(
        tokenizer=tokenizer,
        transform=(
            model.vision_processor if hasattr(model, "vision_processor") else None
        ),
        **asdict(args),
    )
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(
        dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args)
    )

    # print(eval_dataset_dict)
    # loader = data.DataLoader(
    #     # train_dataset,
    #     eval_dataset_dict["ego4d_onlinestep_val"],
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=data_collator,
    #     pin_memory=True
    # )

    # loader = data.DataLoader(
    #     eval_dataset_dict,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=data_collator,
    #     pin_memory=True
    # )

    # print(len(loader))

    # for idx, batch in enumerate(loader):
    #     print(idx)

    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = TrainerWithOnlineGenToEval(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=(
            list(compute_metrics_dict.values())[0]
            if compute_metrics_dict is not None
            else None
        ),
    )
    # save_config(args)
    # trainer.train()

    # trainer.save_model()
    # print("Trained model saved...")

    print("Moving to Evaluation...")
    for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
        loader = data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=simple_data_collator,
            pin_memory=True
        )
        for idx, batch in enumerate(loader):
            conversation, frames, sample_idxs, evaluation_kwargs = batch
            conversation = [
                {"role": "system", "content": args.system_prompt}
            ] + conversation

            ###### add long context here according to the output from the model, check if step exists
            # to-do

            ###### update long term and short term tokens
            # to-do

            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            learn_ranges = (tokenizer.get_learn_ranges(conversation))

            batch = data_processing([text], [frames], [learn_ranges], [sample_idxs], [evaluation_kwargs], tokenizer)
            # print(batch["frames"].size())
            # print(batch["labels"].size())
            # print(batch.input_ids)
            # print(idx)

def save_config(args):
    os.makedirs(args.logging_dir, exist_ok=True)  # Ensure the directory exists
    config_path = os.path.join(args.logging_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(args), f, indent=4)


if __name__ == "__main__":
    train()
