import torch, os, re, json, random, math, copy, itertools, psutil
import Levenshtein as lev
import numpy as np
import pickle as pkl
from transformers import EvalPrediction, PreTrainedTokenizer
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchcodec.decoders import VideoDecoder
from torchaudio.utils import ffmpeg_utils
from typing import Dict, List, Tuple
from collections import defaultdict

# These imports come from pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from ..utils import DictWithTo

from ..stream import StreamMixIn

from .egoclip import EgoClip #,EgoClipOnline

random.seed(42)
np.random.seed(42)
ffmpeg_utils.set_log_level(-8)

import heapq
import itertools
import random

class MinHeapByX:
    def __init__(self):
        self.heap = []
        self.counter = itertools.count()

    def push(self, x, y):
        heapq.heappush(self.heap, (x, next(self.counter), y))

    def pop(self):
        x, _, y = heapq.heappop(self.heap)
        return x, y

    def get_previous_steps(self, x_target, num=5):
        previous_responses = []
        for x, _, y in self.heap:
            if x < x_target and y not in previous_responses: # add previuos step
                previous_responses.append(y)
            if len(previous_responses) == num:
                break
        return previous_responses

    def peek(self):
        if not self.heap:
            return None
        x, _, y = self.heap[0]
        return x, y




class Ego4DOnlinestep(EgoClip, StreamMixIn):
    evaluation_kwargs = DictWithTo(
        evaluator="generate_after_embed",
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        temperature=1.0,
        top_p=1.0,
    )
    crop_with_boxes = None

    user_message = {
        "role": "user",
        "content": "Please output the corresponding action of each frame. If a frame does not show any action, output background. Any previous actions performed are prepended with <L> and interleaved with visual frames.",
    }

    def __init__(
        self,
        *,
        split: str,
        frame_fps: int,
        num_samples: int,
        is_training: bool,
        transform: None,
        **kwargs,
    ):
        super().__init__(
            split=split,
            frame_fps=frame_fps,
            num_samples=num_samples,
            is_training=is_training,
            **kwargs,
        )
        self.split = split
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.anno_fps = 30
        self.num_samples = num_samples
        self.transform = transform
        self.num_frames = 64
        
        print(f"Total {self.split} samples: {len(self.annos)}")

        #### Shih-Po's edition
        # 1. get gt_responses
        # 2. get gt_previous_steps
        self.gt_responses = []
        self.gt_previous_steps = {}

        for i in range(len(self.annos)):
            sample = self.annos.iloc[i]
            response = sample["clip_text"]
            self.gt_responses.append(self.clean_response(response))
            video_start_sec = max(float(sample["clip_start"]), 0)
            if sample["video_uid"] not in self.gt_previous_steps:
                self.gt_previous_steps[sample["video_uid"]] = MinHeapByX()
            
            self.gt_previous_steps[sample["video_uid"]].push(video_start_sec, self.clean_response(response))


    def __getitem__(self, index):
        sample = self.annos.iloc[index]
        video_path, video_sec, bound_sec = self.get_video_path(sample)

        response = sample["clip_text"]
        conversation = [
            self.user_message,
            {"role": "stream"},
            {"role": "assistant", "content": self.clean_response(response)},
        ]

        conversation[-1]["learn"] = True
        conversation[-2]["learn"] = True

        # bboxes, success = self.load_hand_object_bbox(sample)
        #### Shih-Po's edition
        # randomly sample a timestamp in video_sec, and its past frames
        target_sec = random.uniform(video_sec[0], video_sec[1])
        duration = self.num_frames / self.anno_fps
        if video_sec[0] - duration < 0.0:
            video_sec[0] = 0.0
        else:
            video_sec[0] -= duration
        
        video_sec[1] = target_sec

        frames, crop_params, _, seconds = self.get_video_frames(
            video_path,
            video_sec,
            bound_sec,
            num_frames=self.num_frames,
            boxes=None,
            transform=self.transform,
        )
        #### Shih-Po's commnet
        # if duration < num_frames, automatically generates duplicate frames
        ####
        conversation[-2]["num_frames"] = self.num_frames #64 # streaming, at most 64 frames


        # conversation[-2]["long_context"] = ["Open the bag", "Pick up the bag", "asdasd", "zzzzz"] #[""]
        video_start_sec = max(float(sample["clip_start"]), 0)

        ############### debugging for labels
        # when long_context is not empty, the labels are wrong
        conversation[-2]["long_context"] = self.gt_previous_steps[sample["video_uid"]].get_previous_steps(video_start_sec)
        # conversation[-2]["long_context"] = [""]

        conversation = (
            conversation if self.is_training else conversation[:-1]
        )  # if not training, do not include the assistant message

        return (
            *super().__getitem__(
                conversation=conversation,
                load_ranges=frames,
                bboxes=None,
                add_generation_prompt=not self.is_training,
            ),
            index,
            self.evaluation_kwargs,
        )

    def clean_response(self, text: str) -> str:
        # 1. Replace known tags (#C, #O) with intended forms or blanks
        text = text.replace("#C C", "The person")
        text = text.replace("#C ", "the person")
        text = text.replace("#O ", "")
        text = text.replace(".", "")

        # Shih-Po's edition
        text = text.replace("# C C", "The person")
        text = text.replace("#c c", "The person")
        text = text.replace("C C", "The person")

        # 2. Collapse multiple spaces created by removals
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Fix lowercase after sentence boundaries (if tag removal broke capitalization)
        # e.g., "The person is cooking. he is stirring." → "The person is cooking. He is stirring."
        text = re.sub(
            r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

        # 4. Ensure first character is capitalized (if start lost capitalization)
        text = text[0].upper() + text[1:] if text else text

        return text

    def get_video_path(self, sample):
        video_uid = sample["video_uid"]
        video_start_sec = max(float(sample["clip_start"]), 0)
        video_end_sec = max(float(sample["clip_end"]), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(
            self.video_root, video_uid, str(chunk_start_id) + ".mp4"
        )
        full_video_end_fp = os.path.join(
            self.video_root, video_uid, str(chunk_end_id) + ".mp4"
        )

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec

        return video_fp, video_sec, bound_sec

    def load_hand_object_bbox(self, sample):
        clip_start = float(sample["clip_start"])

        hand_boxes = torch.zeros(4, 2, 4)
        obj_boxes = torch.zeros(4, 4, 4)
        image_size = (0, 0)

        video_name = sample["video_uid"]
        clip_index = str(int(clip_start // self.chunk_sec))
        hand_file = os.path.join(
            self.handobj_dir, video_name, clip_index + ".handobj.pkl"
        )
        success = 0
        if os.path.exists(hand_file):
            hand_info = pkl.load(open(hand_file, "rb"))

            poss_starts = [clip_start, clip_start - 0.001, clip_start + 0.001]
            image_size = (
                [*hand_info.values()][0]["info"]["height"],
                [*hand_info.values()][0]["info"]["width"],
            )
            for start in poss_starts:
                try:
                    hand_boxes = torch.stack(
                        [load_bboxes(hand_info[round(start, 3)], i) for i in range(4)]
                    )
                    obj_boxes = torch.stack(
                        [
                            load_bboxes(
                                hand_info[round(start, 3)], i, box_type="obj_dets"
                            )
                            for i in range(4)
                        ]
                    )
                    success = 1
                except:
                    success = 0
                if success == 1:
                    break

        boxes = torch.cat([hand_boxes, obj_boxes], 1)
        boxes = boxes / torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0]],
            dtype=boxes.dtype,
            device=boxes.device,
        )

        if np.isnan(boxes).any():
            hand_boxes = torch.zeros(4, 2, 4)
            obj_boxes = torch.zeros(4, 4, 4)
            boxes = torch.cat([hand_boxes, obj_boxes], 1)
            success = 0
            # print("Found Nan bbox")
            # print(boxes)
        
        return boxes, success

    def get_video_frames(
        self, video_fp, video_sec, bound_sec, num_frames=4, boxes=None, pred=False, transform=False
    ):
        video_params = {"input_res": 224, "num_frames": num_frames, "loading": "lax"}
        video_loading = video_params.get("loading", "loose")
        try:
            # assert False
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                images, seconds = read_frames_cv2_egoclip_torchcodec(
                    video_fp[0],
                    video_sec[0],
                    end_second=video_sec[1],
                    clip_length=video_params["num_frames"],
                )
                valid = 1
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                img = Image.new(
                    "RGB",
                    (video_params["input_res"], video_params["input_res"]),
                    (0, 0, 0),
                )
                images = [img.copy() for _ in range(video_params["num_frames"])]
                valid = 0
                seconds = [0.0] * video_params["num_frames"]
                print(f"[WARN] torchcodec failed for {video_fp}: {e}")

        crop_params = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if transform:
            if hasattr(transform, "tokenizer"):
                frames = transform(
                    text="A", images=images, return_tensors="pt", padding=True
                )["pixel_values"]
            else:
                frames = transform(images=images, return_tensors="pt")["pixel_values"]
        else:
            raise ValueError("Transform must be provided for video frames.")

        return frames, crop_params, valid, seconds
    

    def prepare_data_for_coco_eval(
        self,
        predictions: Dict[str, str],
        references: Dict[str, List[str]],
    ):
        """
        Convert dictionaries into the format expected by pycocoevalcap:
        - gts: dict[image_id] -> list of reference strings
        - res: dict[image_id] -> list of hypothesis strings (usually len==1)
        """
        gts = {}
        res = {}

        for img_id, refs in references.items():
            # ground truths: list of reference captions
            gts[img_id] = [r.strip() for r in refs]

        for img_id, pred in predictions.items():
            # result: list containing a single prediction string
            res[img_id] = [pred.strip()]

        return gts, res

    ### Shih-Po's edition
    def compute_bleu_cider_spice(self, predictions, references):
        """
        Compute BLEU-4, CIDEr, SPICE over the dataset.
        Returns a dict with:
        - 'BLEU-4'
        - 'CIDEr'
        - 'SPICE'
        """
        preds, res = self.prepare_data_for_coco_eval(predictions, references)
        # ----------------------
        # BLEU (1-4)
        # ----------------------
        bleu_scorer = Bleu(n=4)
        bleu_scores, _ = bleu_scorer.compute_score(preds, res)
        # bleu_scores is a list of scores for BLEU-1,2,3,4

        # ----------------------
        # CIDEr
        # ----------------------
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(preds, res)

        # ----------------------
        # SPICE, which needs java
        # ----------------------
        # spice_scorer = Spice()
        # spice_score, spice_scores_per_image = spice_scorer.compute_score(preds, res)

        metrics = {
            "BLEU-4": bleu_scores[3] * 100,
            "CIDEr": cider_score * 100,
            # "SPICE": spice_score * 100,
        }

        return metrics
    
    ### Shih-Po's edition, from egoexo4d
    def compute_metrics(
        self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs
    ):
        out_dir = kwargs["output_dir"]
        batch_pred_tensor, sample_idxs = (
            eval_predictions.predictions,
            eval_predictions.label_ids,
        )
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id
        predictions = tokenizer.batch_decode(
            batch_pred_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        with open(f"{out_dir}/preds.txt", "w") as f_pred:
            f_pred.write("\n".join(predictions))

        with open(f"{out_dir}/gts.txt", "w") as f_gt:
            f_gt.write("\n".join(self.gt_responses))

        dict_references = {}
        dict_predictions = {}
        i = 1
        for pred, ref in zip(predictions, self.gt_responses):
            dict_predictions[i] = pred
            dict_references[i] = [ref]
            i += 1

        output_json = self.compute_bleu_cider_spice(dict_predictions, dict_references)

        with open(f"{out_dir}/eval_results.json", "w") as f_output:
            json.dump(output_json, f_output)
        
        return output_json


import glob, json, math, os, re, torch, tqdm
import pandas as pd

class Ego4DOnlinestep_val(torch.utils.data.Dataset):
    evaluation_kwargs = DictWithTo(
        evaluator="generate_after_embed",
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        temperature=1.0,
        top_p=1.0,
    )
    crop_with_boxes = None
    ignore = {
        "c9c9d2d2-f9cb-405b-b5ab-f48ecf988aaa.mp4",
        "f130564b-a153-4a84-96bf-447cb837272c.mp4",
        "ad71a786-60a3-4be1-b2fb-fc714e061115.mp4",
        "e94b366a-a6dc-4433-a409-c36567ac6ba9.mp4",
        "63a85af7-e27d-438e-90c8-f416efcfb36c.mp4",
        "ecd0d190-ea38-4731-af01-96e05a27ab79.mp4",
        "fb45b5e3-498c-4177-a7f4-161063093014.mp4",
        "0f14d5fb-d911-48aa-9c0d-6bcd10427742.mp4",
        "0e7ba211-0dba-40b8-8ace-a3e5932db4fb.mp4",
        "cb5f5863-555f-495c-badc-f3c29828c1b0.mp4",
        "158629ec-b436-4edb-bcdf-60b4fed8674d.mp4",
        "19dd53a5-1a7f-4342-9849-f251006058af.mp4",
        "6a75b089-b74e-4e45-a345-422587f04f01.mp4",
        # the videos below are shorter than the annotation
        "a24e2240-8720-4bc2-aad8-0801f0a2c5e6.mp4",
        "a25ce65f-5e53-4154-869d-ec61d2a6a9a8.mp4",
        "a2437493-31b9-4574-9a35-3bdb8f38196d.mp4",
        "9d26362a-4411-4d8b-b114-03f4f89928d7.mp4",
        "9d26362a-4411-4d8b-b114-03f4f89928d7.mp4",
        "9e9aa4f5-e15b-412f-b84d-3e97b2d1f08b.mp4",
        "9d209484-b083-4731-b5dc-d3af3df09292.mp4",
        "9e9c5b05-c7d4-450d-9850-e6ae83caa9a5.mp4",
        "a094dc09-0db0-4ccd-bfde-ff1001db4d3e.mp4",
        "2665edd9-4ab5-493b-aa3d-dd156be107a7.mp4",
        "b6d64c01-1462-4ebf-9eb5-44585cdd8ac4.mp4",
        "9d0c46d6-ea9a-43e6-8b8a-4b2972eb6ef4.mp4",
        "5c3fdf43-280d-4b06-8061-cf68e268f367.mp4",
        "0a74808e-4f55-4bd4-abbd-78f3435ea5bc.mp4", # does not exist
        "864a2391-63d5-4f64-9ba2-cf1367c178c2.mp4", # does not exist
        "4d29d4f8-7bcc-45df-9182-0937b462d7a1.mp4", # does not exist
        "d0fe52fc-cb32-4bcc-ac7c-0312968a8b98.mp4", # does not exist
        "67cf6d70-7387-45ab-8200-27ce803476f8.mp4", # does not exist
        "7a39a702-dcd1-46c4-82aa-00c97bda423e.mp4", # does not exist
        "29bc686e-8f5c-49de-a23e-c0b802e47d4d.mp4", # does not exist
    }
    user_message = {
        "role": "user",
        "content": "Please output the corresponding action of each frame. If a frame does not show any action, output background. Any previous actions performed are prepended with <L> and interleaved with visual frames.",
    }

    def __init__(
        self,
        *,
        split: str,
        dataset_dir: str,
        frame_fps: int,
        num_samples: int,
        is_training: bool,
        transform: None,
        **kwargs,
    ):
        super().__init__()

        self.root = dataset_dir

        self.video_root = os.path.join(self.root, "egoclip/video_chunks")
        self.anno_root = os.path.join(self.root, "egoclip/helping_hands")

        self.frame_fps = frame_fps

        # assert split == "train", "EgoClip only supports Stage-1 pretraining."
        self.split = split

        self.chunk_sec = 600  # each video chunk is 10min long
        self.annos, self.annos_by_segment_id, self.handobj_dir = self.get_metadata()


        self.split = split
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.anno_fps = 30
        self.num_samples = num_samples
        self.transform = transform
        self.num_frames = 64
        
        print(f"Total {self.split} samples: {len(self.annos)}")

        #### Shih-Po's edition
        # 1. get gt_responses
        # 2. get gt_previous_steps
        self.gt_responses = []
        self.gt_previous_steps = {}

        for i in range(len(self.annos)):
            sample = self.annos.iloc[i]
            response = sample["clip_text"][0]
            self.gt_responses.append(self.clean_response(response))
            video_start_sec = max(float(sample["clip_start"]), 0)
            if sample["video_uid"] not in self.gt_previous_steps:
                self.gt_previous_steps[sample["video_uid"]] = MinHeapByX()
            
            self.gt_previous_steps[sample["video_uid"]].push(video_start_sec, self.clean_response(response))

    def __len__(self):
        return len(self.annos)

    def get_metadata(self):
        split_files = {
            #"train": "egoclip.csv",
            # "train": "clean_egoclip.csv",
            ################ for debugging
            "train": "tiny_val_egoclip.csv",            
            # "val": "online_val_egoclip.csv",
            "val": "online_tiny_val_egoclip.csv",
            # "val": "val_egoclip.csv",
        }
        file = split_files[self.split]

        meta_dir = os.path.join(self.anno_root, "metadata/EgoClip")
        handobj_dir = os.path.join(
            self.anno_root, "hand_object_clip_per_video_4f_lavila_narrator_640"
        )

        annos = pd.read_csv(
            os.path.join(meta_dir, file),
            sep="\t",
            on_bad_lines="skip",
        )

        # ignore bad videos
        annos = annos[~annos["video_uid"].astype(str).add(".mp4").isin(self.ignore)]

        annos["segment_id"] = (
            annos["video_uid"]
            + "_"
            + (annos["narration_time"] // self.chunk_sec).astype(str)
        )
        annos_by_segment_id = dict(tuple(annos.groupby("segment_id")))
        print("!!! EgoClip metadata loaded...")

        return annos, annos_by_segment_id, handobj_dir

    def __getitem__(self, index):
        sample = self.annos.iloc[index]
        video_path, video_sec, bound_sec = self.get_video_path(sample)

        response = sample["clip_text"][0] # select the first response as GT
        conversation = [
            self.user_message,
            {"role": "stream"},
            {"role": "assistant", "content": self.clean_response(response)},
        ]

        conversation[-1]["learn"] = True
        conversation[-2]["learn"] = True

        # bboxes, success = self.load_hand_object_bbox(sample)

        frames, crop_params, _, seconds = self.get_video_frames(
            video_path,
            video_sec,
            bound_sec,
            num_frames=self.num_frames,
            boxes=None,
            transform=self.transform,
        )
        #### Shih-Po's commnet
        # if duration < num_frames, automatically generates duplicate frames
        ####
        conversation[-2]["num_frames"] = self.num_frames #64 # streaming, at most 64 frames


        # conversation[-2]["long_context"] = ["Open the bag", "Pick up the bag", "asdasd", "zzzzz"] #[""]
        # video_start_sec = max(float(sample["clip_start"]), 0)

        ############### debugging for labels
        # when long_context is not empty, the labels are wrong
        # conversation[-2]["long_context"] = self.gt_previous_steps[sample["video_uid"]].get_previous_steps(video_start_sec)
        conversation[-2]["long_context"] = [""]

        conversation = (
            conversation if self.is_training else conversation[:-1]
        )  # if not training, do not include the assistant message

        return conversation, frames
        # return (
        #     *super().__getitem__(
        #         conversation=conversation,
        #         load_ranges=frames,
        #         bboxes=None,
        #         add_generation_prompt=not self.is_training,
        #     ),
        #     index,
        #     self.evaluation_kwargs,
        # )

    def clean_response(self, text: str) -> str:
        # 1. Replace known tags (#C, #O) with intended forms or blanks
        text = text.replace("#C C", "The person")
        text = text.replace("#C ", "the person")
        text = text.replace("#O ", "")
        text = text.replace(".", "")

        # Shih-Po's edition
        text = text.replace("# C C", "The person")
        text = text.replace("#c c", "The person")
        text = text.replace("C C", "The person")

        # 2. Collapse multiple spaces created by removals
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Fix lowercase after sentence boundaries (if tag removal broke capitalization)
        # e.g., "The person is cooking. he is stirring." → "The person is cooking. He is stirring."
        text = re.sub(
            r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

        # 4. Ensure first character is capitalized (if start lost capitalization)
        text = text[0].upper() + text[1:] if text else text

        return text

    def get_video_path(self, sample):
        video_uid = sample["video_uid"]
        video_start_sec = max(float(sample["clip_start"]), 0)
        video_end_sec = max(float(sample["clip_end"]), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(
            self.video_root, video_uid, str(chunk_start_id) + ".mp4"
        )
        full_video_end_fp = os.path.join(
            self.video_root, video_uid, str(chunk_end_id) + ".mp4"
        )

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec

        return video_fp, video_sec, bound_sec

    def load_hand_object_bbox(self, sample):
        clip_start = float(sample["clip_start"])

        hand_boxes = torch.zeros(4, 2, 4)
        obj_boxes = torch.zeros(4, 4, 4)
        image_size = (0, 0)

        video_name = sample["video_uid"]
        clip_index = str(int(clip_start // self.chunk_sec))
        hand_file = os.path.join(
            self.handobj_dir, video_name, clip_index + ".handobj.pkl"
        )
        success = 0
        if os.path.exists(hand_file):
            hand_info = pkl.load(open(hand_file, "rb"))

            poss_starts = [clip_start, clip_start - 0.001, clip_start + 0.001]
            image_size = (
                [*hand_info.values()][0]["info"]["height"],
                [*hand_info.values()][0]["info"]["width"],
            )
            for start in poss_starts:
                try:
                    hand_boxes = torch.stack(
                        [load_bboxes(hand_info[round(start, 3)], i) for i in range(4)]
                    )
                    obj_boxes = torch.stack(
                        [
                            load_bboxes(
                                hand_info[round(start, 3)], i, box_type="obj_dets"
                            )
                            for i in range(4)
                        ]
                    )
                    success = 1
                except:
                    success = 0
                if success == 1:
                    break

        boxes = torch.cat([hand_boxes, obj_boxes], 1)
        boxes = boxes / torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0]],
            dtype=boxes.dtype,
            device=boxes.device,
        )

        if np.isnan(boxes).any():
            hand_boxes = torch.zeros(4, 2, 4)
            obj_boxes = torch.zeros(4, 4, 4)
            boxes = torch.cat([hand_boxes, obj_boxes], 1)
            success = 0
            # print("Found Nan bbox")
            # print(boxes)
        
        return boxes, success

    def get_video_frames(
        self, video_fp, video_sec, bound_sec, num_frames=4, boxes=None, pred=False, transform=False
    ):
        video_params = {"input_res": 224, "num_frames": num_frames, "loading": "lax"}
        video_loading = video_params.get("loading", "loose")
        try:
            # assert False
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                images, seconds = read_frames_cv2_egoclip_torchcodec(
                    video_fp[0],
                    video_sec[0],
                    end_second=video_sec[1],
                    clip_length=video_params["num_frames"],
                )
                valid = 1
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                img = Image.new(
                    "RGB",
                    (video_params["input_res"], video_params["input_res"]),
                    (0, 0, 0),
                )
                images = [img.copy() for _ in range(video_params["num_frames"])]
                valid = 0
                seconds = [0.0] * video_params["num_frames"]
                print(f"[WARN] torchcodec failed for {video_fp}: {e}")

        crop_params = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if transform:
            if hasattr(transform, "tokenizer"):
                frames = transform(
                    text="A", images=images, return_tensors="pt", padding=True
                )["pixel_values"]
            else:
                frames = transform(images=images, return_tensors="pt")["pixel_values"]
        else:
            raise ValueError("Transform must be provided for video frames.")

        return frames, crop_params, valid, seconds
    

    def prepare_data_for_coco_eval(
        self,
        predictions: Dict[str, str],
        references: Dict[str, List[str]],
    ):
        """
        Convert dictionaries into the format expected by pycocoevalcap:
        - gts: dict[image_id] -> list of reference strings
        - res: dict[image_id] -> list of hypothesis strings (usually len==1)
        """
        gts = {}
        res = {}

        for img_id, refs in references.items():
            # ground truths: list of reference captions
            gts[img_id] = [r.strip() for r in refs]

        for img_id, pred in predictions.items():
            # result: list containing a single prediction string
            res[img_id] = [pred.strip()]

        return gts, res

    ### Shih-Po's edition
    def compute_bleu_cider_spice(self, predictions, references):
        """
        Compute BLEU-4, CIDEr, SPICE over the dataset.
        Returns a dict with:
        - 'BLEU-4'
        - 'CIDEr'
        - 'SPICE'
        """
        preds, res = self.prepare_data_for_coco_eval(predictions, references)
        # ----------------------
        # BLEU (1-4)
        # ----------------------
        bleu_scorer = Bleu(n=4)
        bleu_scores, _ = bleu_scorer.compute_score(preds, res)
        # bleu_scores is a list of scores for BLEU-1,2,3,4

        # ----------------------
        # CIDEr
        # ----------------------
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(preds, res)

        # ----------------------
        # SPICE, which needs java
        # ----------------------
        # spice_scorer = Spice()
        # spice_score, spice_scores_per_image = spice_scorer.compute_score(preds, res)

        metrics = {
            "BLEU-4": bleu_scores[3] * 100,
            "CIDEr": cider_score * 100,
            # "SPICE": spice_score * 100,
        }

        return metrics
    
    ### Shih-Po's edition, from egoexo4d
    def compute_metrics(
        self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs
    ):
        out_dir = kwargs["output_dir"]
        batch_pred_tensor, sample_idxs = (
            eval_predictions.predictions,
            eval_predictions.label_ids,
        )
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id
        predictions = tokenizer.batch_decode(
            batch_pred_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        with open(f"{out_dir}/preds.txt", "w") as f_pred:
            f_pred.write("\n".join(predictions))

        with open(f"{out_dir}/gts.txt", "w") as f_gt:
            f_gt.write("\n".join(self.gt_responses))

        dict_references = {}
        dict_predictions = {}
        i = 1
        for pred, ref in zip(predictions, self.gt_responses):
            dict_predictions[i] = pred
            dict_references[i] = [ref]
            i += 1

        output_json = self.compute_bleu_cider_spice(dict_predictions, dict_references)

        with open(f"{out_dir}/eval_results.json", "w") as f_output:
            json.dump(output_json, f_output)
        
        return output_json

def build_ego4d_onlinestep_train(**kwargs):
    return Ego4DOnlinestep(split="train", **kwargs)

def build_ego4d_onlinestep_val(**kwargs):
    return Ego4DOnlinestep_val(split="val", **kwargs)

def load_bboxes(hand_info, ind, box_type="hand_dets", threshold=0.5):
    ind = ind % 600
    if box_type == "hand_dets":
        max_boxes = 2
    else:
        max_boxes = 4
    out_boxes = torch.zeros(max_boxes, 4)
    if int(ind) in hand_info:
        dets = hand_info[int(ind)][box_type]
        if dets is not None:
            boxes = torch.tensor(dets[:, :4])
            scores = torch.tensor(dets[:, 4:5])[:, 0]
            boxes, scores = boxes[scores > 0.5], scores[scores > 0.5]
            topk = torch.argsort(scores, descending=True)[:max_boxes]
            out_boxes[: len(topk)] = boxes[topk]
    return out_boxes


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def read_frames_cv2_egoclip_torchcodec(
    vpath,
    start_second,
    end_second=None,
    chunk_len=600,
    fps=30,
    clip_length=32,
    jitter=False,
):
    dec = VideoDecoder(vpath, device="cpu", dimension_order="NHWC")
    frame_count = dec.metadata.num_frames
    stream_fps = dec.metadata.average_fps

    eff_fps = float(stream_fps) if (fps == -1 and stream_fps) else float(fps)
    if eff_fps <= 0:
        eff_fps = 30.0

    duration_guess = None
    if frame_count is not None and frame_count > 0:
        duration_guess = frame_count / eff_fps

    if chunk_len == -1:
        # whole video; clamp end to duration if known
        if end_second is None:
            end_second = (
                len(dec) / eff_fps
                if frame_count is not None
                else (start_second + clip_length / eff_fps)
            )
        if duration_guess is not None:
            end_second = min(end_second, duration_guess)
        second_offset = float(start_second)  # local == global
    else:
        chunk_start = (int(start_second) // int(chunk_len)) * int(chunk_len)
        second_offset = float(start_second) - float(chunk_start)
        if end_second is None:
            end_second = start_second + (clip_length / eff_fps)

    frame_offset = int(np.round(second_offset * eff_fps))
    total_duration = max(int((end_second - start_second) * eff_fps), clip_length)

    if chunk_len == -1:
        if end_second <= start_second:
            raise ValueError("end_second should be greater than start_second")
        start_idx = frame_offset
        end_idx_excl = min(
            frame_offset + total_duration,
            len(dec) if frame_count is not None else (frame_offset + total_duration),
        )
    else:
        start_idx = frame_offset
        end_idx_excl = frame_offset + total_duration

    frame_ids = get_frame_ids(
        start_idx,
        end_idx_excl,
        num_segments=clip_length,
        jitter=jitter,
    )

    # --- decode exact indices; handle tail overflow by duplicating last valid frame ---
    n = len(dec) if frame_count is not None else None
    # split indices into valid and overflow
    if n is not None:
        valid_ids = [i for i in frame_ids if 0 <= i < n]
        overflow_count = len(frame_ids) - len(valid_ids)
    else:
        valid_ids = frame_ids
        overflow_count = 0

    try:
        frames_t = dec.get_frames_at(valid_ids).data
    except Exception as e:
        chunks = []
        for i in valid_ids:
            try:
                chunks.append(dec.get_frames_at([i]).data)  # shape [1,H,W,C]
            except Exception:
                pass
        if not chunks:
            raise
        frames_t = torch.cat(chunks, dim=0)  # [V,H,W,C]

    t_got = frames_t.shape[0]
    if t_got < clip_length:
        if t_got == 0:
            H = W = 224
            frames_t = torch.zeros((1, H, W, 3), dtype=torch.uint8)
            t_got = 1
        pad = frames_t[-1:].repeat(clip_length - t_got, 1, 1, 1)
        frames_t = torch.cat([frames_t, pad], dim=0)

    frames = frames_t.cpu().numpy()  # [T,H,W,C], uint8
    times = [f / 30.0 for f in frame_ids]

    return frames, times
