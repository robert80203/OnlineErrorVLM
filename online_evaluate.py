import json
import os
from dataclasses import asdict
import torch.utils.data as data


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
    batch_conversation, batch_frames = batch
    return batch_conversation[0], batch_frames[0]

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
            print(idx)
            conversation, frames = batch
            conversation = [
                {"role": "system", "content": args.system_prompt}
            ] + conversation

            print(conversation)

            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            learn_ranges = (tokenizer.get_learn_ranges(conversation))

            print(text)


def save_config(args):
    os.makedirs(args.logging_dir, exist_ok=True)  # Ensure the directory exists
    config_path = os.path.join(args.logging_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(args), f, indent=4)


if __name__ == "__main__":
    train()
