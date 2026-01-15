import glob, json, math, os, re, torch, tqdm
import pandas as pd


class EgoClip:
    # root = "/scr/shihpo" #"datasets/ego4d"
    # video_root = os.path.join(root, "videos")
    # anno_root = os.path.join(root, "annotations")

    # please put videos that could not be downloaded, or could not be opened with torchcodec
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

    def __init__(
        self,
        split: str,
        vision_pretrained: list,
        dataset_dir: str,
        frame_fps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.root = os.path.join(dataset_dir, "ego4d")
        self.root = dataset_dir

        self.video_root = os.path.join(self.root, "egoclip/video_chunks")
        self.anno_root = os.path.join(self.root, "egoclip/helping_hands")

        self.frame_fps = frame_fps

        # assert split == "train", "EgoClip only supports Stage-1 pretraining."
        self.split = split

        self.chunk_sec = 600  # each video chunk is 10min long
        self.annos, self.annos_by_segment_id, self.handobj_dir = self.get_metadata()

    def __len__(self):
        return len(self.annos)

    def get_metadata(self):
        split_files = {
            #"train": "egoclip.csv",
            # "train": "clean_egoclip.csv",
            ################ for debugging
            "train": "tiny_val_egoclip.csv",

            
            "val": "tiny_val_egoclip.csv",
            # "val": "val_egoclip.csv",
        }
        # split_files = {
        #     "train": "debug_egoclip.csv",
        #     "val": "egomcq.json",
        #     "test": "egomcq.json",
        # }
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


# class EgoClipOnline:
#     # please put videos that could not be downloaded, or could not be opened with torchcodec
#     ignore = {
#         "c9c9d2d2-f9cb-405b-b5ab-f48ecf988aaa.mp4",
#         "f130564b-a153-4a84-96bf-447cb837272c.mp4",
#         "ad71a786-60a3-4be1-b2fb-fc714e061115.mp4",
#         "e94b366a-a6dc-4433-a409-c36567ac6ba9.mp4",
#         "63a85af7-e27d-438e-90c8-f416efcfb36c.mp4",
#         "ecd0d190-ea38-4731-af01-96e05a27ab79.mp4",
#         "fb45b5e3-498c-4177-a7f4-161063093014.mp4",
#         "0f14d5fb-d911-48aa-9c0d-6bcd10427742.mp4",
#         "0e7ba211-0dba-40b8-8ace-a3e5932db4fb.mp4",
#         "cb5f5863-555f-495c-badc-f3c29828c1b0.mp4",
#         "158629ec-b436-4edb-bcdf-60b4fed8674d.mp4",
#         "19dd53a5-1a7f-4342-9849-f251006058af.mp4",
#         "6a75b089-b74e-4e45-a345-422587f04f01.mp4",
#         # the videos below are shorter than the annotation
#         "a24e2240-8720-4bc2-aad8-0801f0a2c5e6.mp4",
#         "a25ce65f-5e53-4154-869d-ec61d2a6a9a8.mp4",
#         "a2437493-31b9-4574-9a35-3bdb8f38196d.mp4",
#         "9d26362a-4411-4d8b-b114-03f4f89928d7.mp4",
#         "9d26362a-4411-4d8b-b114-03f4f89928d7.mp4",
#         "9e9aa4f5-e15b-412f-b84d-3e97b2d1f08b.mp4",
#         "9d209484-b083-4731-b5dc-d3af3df09292.mp4",
#         "9e9c5b05-c7d4-450d-9850-e6ae83caa9a5.mp4",
#         "a094dc09-0db0-4ccd-bfde-ff1001db4d3e.mp4",
#         "2665edd9-4ab5-493b-aa3d-dd156be107a7.mp4",
#         "b6d64c01-1462-4ebf-9eb5-44585cdd8ac4.mp4",
#         "9d0c46d6-ea9a-43e6-8b8a-4b2972eb6ef4.mp4",
#         "5c3fdf43-280d-4b06-8061-cf68e268f367.mp4",
#         "0a74808e-4f55-4bd4-abbd-78f3435ea5bc.mp4", # does not exist
#         "864a2391-63d5-4f64-9ba2-cf1367c178c2.mp4", # does not exist
#         "4d29d4f8-7bcc-45df-9182-0937b462d7a1.mp4", # does not exist
#         "d0fe52fc-cb32-4bcc-ac7c-0312968a8b98.mp4", # does not exist
#         "67cf6d70-7387-45ab-8200-27ce803476f8.mp4", # does not exist
#         "7a39a702-dcd1-46c4-82aa-00c97bda423e.mp4", # does not exist
#         "29bc686e-8f5c-49de-a23e-c0b802e47d4d.mp4", # does not exist
#     }

#     def __init__(
#         self,
#         split: str,
#         vision_pretrained: list,
#         dataset_dir: str,
#         frame_fps: int,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         # self.root = os.path.join(dataset_dir, "ego4d")
#         self.root = dataset_dir

#         self.video_root = os.path.join(self.root, "egoclip/video_chunks")
#         self.anno_root = os.path.join(self.root, "egoclip/helping_hands")

#         self.frame_fps = frame_fps

#         # assert split == "train", "EgoClip only supports Stage-1 pretraining."
#         self.split = split

#         self.chunk_sec = 600  # each video chunk is 10min long
#         self.annos, self.annos_by_segment_id, self.handobj_dir = self.get_metadata()

#     def __len__(self):
#         return len(self.annos)

#     def get_metadata(self):
#         split_files = {
#             #"train": "egoclip.csv",
#             # "train": "clean_egoclip.csv",
#             ################ for debugging
#             "train": "tiny_val_egoclip.csv",            
#             # "val": "online_val_egoclip.csv",
#             "val": "online_tiny_val_egoclip.csv",
#             # "val": "val_egoclip.csv",
#         }
#         file = split_files[self.split]

#         meta_dir = os.path.join(self.anno_root, "metadata/EgoClip")
#         handobj_dir = os.path.join(
#             self.anno_root, "hand_object_clip_per_video_4f_lavila_narrator_640"
#         )

#         annos = pd.read_csv(
#             os.path.join(meta_dir, file),
#             sep="\t",
#             on_bad_lines="skip",
#         )

#         # ignore bad videos
#         annos = annos[~annos["video_uid"].astype(str).add(".mp4").isin(self.ignore)]

#         annos["segment_id"] = (
#             annos["video_uid"]
#             + "_"
#             + (annos["narration_time"] // self.chunk_sec).astype(str)
#         )
#         annos_by_segment_id = dict(tuple(annos.groupby("segment_id")))
#         print("!!! EgoClip metadata loaded...")

#         return annos, annos_by_segment_id, handobj_dir