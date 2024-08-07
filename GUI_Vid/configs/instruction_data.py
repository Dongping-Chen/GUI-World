import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "<your path to training data>"

# ============== pretraining datasets=================
available_corpus = dict(
    # Images
    mobile_gui_image_concise_caption=[
        f"/media/sata4/final_videos/meta_gui_image/annotations/train_concise_caption_data.json", 
        "/media/sata4/final_videos/",
    ],
    mobile_gui_image_detailed_caption=[
        f"/media/sata4/final_videos/meta_gui_image/annotations/train_detailed_caption_data.json", 
        "/media/sata4/final_videos/",
    ],
    omniact_image_concise_caption=[
        "/media/sata4/final_videos/omniact/concise_caption_data.json", 
        "/media/sata4/",
    ],
    omniact_image_detailed_caption=[
        "/media/sata4/final_videos/omniact/detailed_caption_data.json", 
        "/media/sata4/",
    ],
    gui_image_concise_caption=[
        f"{anno_root_it}/v3_new_concise_caption_data.json", 
        "/media/sata4/",
    ],
    gui_image_detailed_caption=[
        f"{anno_root_it}/v3_new_detailed_caption_data.json", 
        "/media/sata4/",
    ],
    gui_video_caption=[
        f"{anno_root_it}/v3_new_caption_data.json",
        "/media/sata4/",
        "video"
    ],
    # Videos
    gui_video_short_caption=[
        f"{anno_root_it}/v3_new_short_caption_data.json",
        "/media/sata4/",
        "video"
    ],
    gui_video_reasoning=[
        f"{anno_root_it}/v3_new_reasoning_data.json",
        "/media/sata4/",
        "video"
    ],
    gui_video_vqa=[
        f"{anno_root_it}/v3_new_vqa_data.json",
        "/media/sata4/",
        "video"
    ],
    gui_video_conversation=[
        f"{anno_root_it}/v3_new_conversation_data.json",
        "/media/sata4/",
        "video"
    ]
)


# add mc for clevrer_qa
available_corpus["videochat2_instruction"] = [
    # Images
    # available_corpus['mobile_gui_image_concise_caption'],
    # available_corpus['mobile_gui_image_detailed_caption'],
    # available_corpus['omniact_image_concise_caption'],
    # available_corpus['omniact_image_detailed_caption'],
    # available_corpus['gui_image_detailed_caption'],
    # available_corpus['gui_image_concise_caption'],
    
    # Videos
    available_corpus['gui_video_caption'],
    available_corpus['gui_video_short_caption'],
    available_corpus['gui_video_reasoning'],
    available_corpus['gui_video_vqa'],
    available_corpus['gui_video_conversation']
]
