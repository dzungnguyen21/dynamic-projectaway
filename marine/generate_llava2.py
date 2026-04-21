import argparse
import torch
import os
import json
import shortuuid

from torch.utils.data import DataLoader
from transformers import LogitsProcessorList
from tqdm.auto import tqdm

from marine.utils.utils import get_chunk, get_answers_file_name, get_model_name_from_path
from marine.utils.utils_dataset import COCOEvalDataset, custom_collate_fn
from marine.utils.utils_guidance import GuidanceLogits
from marine.utils.utils_model import load_model

# Dynamic-PROJECTAWAY (imported lazily to keep 'marine' method import-free)
_DPA_CLASS = None

def _get_dpa_class():
    global _DPA_CLASS
    if _DPA_CLASS is None:
        from marine.utils.dynamic_projectaway import DynamicProjectAway
        _DPA_CLASS = DynamicProjectAway
    return _DPA_CLASS


def _parse_layer_range(s: str):
    """Parse a layer range string like '19-26' into (19, 26)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Layer range must be in 'start-end' format, got: {s}")
    return int(parts[0]), int(parts[1])


def eval_model(args):
    
    # Model
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    
    model, tokenizer, processor = load_model(model_name, model_path, method=args.method)

    # QA Data
    questions = json.load(open(os.path.expanduser(
        os.path.join(args.question_path, args.question_file)), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if args.answers_file is None:
        args.answers_file = get_answers_file_name(args, model_name)

    answers_file = os.path.expanduser(
        os.path.join(args.answer_path, args.answers_file))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = COCOEvalDataset(questions, args.image_folder, processor, tokenizer, args.conv_mode, getattr(model.config, 'mm_use_im_start_end', False))
    eval_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    total_batches = len(eval_dataloader)
    total_samples = len(dataset)

    # Initialize M3ID engine if selected
    if args.method == "m3id_plus":
        from marine.utils.m3id_plus import LLaVA_M3ID_Plus, GammaNet
        gamma_net = GammaNet(hidden_dim=4096)
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gamma_net_weights_full.pth")
        if os.path.exists(weights_path):
            gamma_net.load_state_dict(torch.load(weights_path))
        else:
            # Fallback to local execution ref
            if os.path.exists("gamma_net_weights_full.pth"):
                gamma_net.load_state_dict(torch.load("gamma_net_weights_full.pth"))
                
        engine = LLaVA_M3ID_Plus(model, processor, gamma_net)
        
    # generate
    pbar = tqdm(eval_dataloader, total=total_batches,
                desc=f"Generating [gs={args.guidance_strength}]",
                unit="batch", dynamic_ncols=True)
    samples_done = 0
    for prompts, question_ids, img_ids, input_ids, guidance_ids, images, guidance_images, attention_masks, guidance_attention_masks in pbar:
        
        with torch.inference_mode():
            if args.guidance_strength == 0:
                output_ids = model.generate(
                    input_ids,
                    pixel_values=images,
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
            elif args.method == "dynamic_projectaway":
                # Build a DynamicProjectAway processor for this batch.
                # PA vectors are extracted once per batch via a guidance forward pass.
                dpa = _get_dpa_class()(
                    model=model,
                    guidance_ids=guidance_ids,
                    guidance_images=guidance_images,
                    guidance_attn_mask=guidance_attention_masks,
                    input_ids=input_ids,
                    hgai_layer_range=_parse_layer_range(args.hgai_layers),
                    pa_layer_range=_parse_layer_range(args.pa_layers),
                    dilution_threshold=args.dilution_threshold,
                    hgai_amplify_factor=args.hgai_amplify_factor,
                    sink_threshold=args.sink_threshold,
                    guidance_strength=args.guidance_strength,
                    tokenizer=tokenizer,
                )
                try:
                    output_ids = model.generate(
                        input_ids,
                        pixel_values=images,
                        do_sample=args.sampling,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        logits_processor=LogitsProcessorList([dpa]),
                    )
                finally:
                    # Remove hooks regardless of generation outcome to prevent
                    # hook accumulation across batches.
                    dpa.cleanup()
            elif args.method == "m3id_plus":
                decoded_outputs = []
                for i in range(len(prompts)):
                    raw_prompt = prompts[i].replace("<image>\n", "").strip()
                    image_path = os.path.join(args.image_folder, img_ids[i])
                    response = engine.generate(prompt=raw_prompt, image_path=image_path, max_new_tokens=args.max_new_tokens, visualize=False)
                    decoded_outputs.append(response)
            else:
                # Original MARINE: classifier-free guidance at logit level
                output_ids = model.generate(
                    input_ids,
                    pixel_values=images,
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([
                        GuidanceLogits(guidance_strength=args.guidance_strength,
                                  guidance=guidance_ids,
                                  images=guidance_images,
                                  attention_mask=guidance_attention_masks,
                                  model=model,
                                  tokenizer=tokenizer),
                    ])
                )

        input_token_len = input_ids.shape[1]

        if args.guidance_strength == 0 or args.method != "m3id_plus":
            # Batch decode the outputs
            decoded_outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True)

        samples_done += len(decoded_outputs)
        pbar.set_postfix({"samples": f"{samples_done}/{total_samples}"})

        for i, output in enumerate(decoded_outputs):

            # Process each output
            output = output.strip()

            # Generate answer ID and write to file
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": question_ids[i],
                                       "image_id": img_ids[i],
                                       "prompt": prompts[i],
                                       "text": output,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")

        ans_file.flush()
    ans_file.close()
    print(f"Done! Saved answers to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_folder", type=str,
                        default="./data/coco/val2014")
    parser.add_argument("--question_path", type=str,
                        default="./data/marine_qa/question")
    parser.add_argument("--question_file", type=str,
                        default="I02_mmc4_grey_ram_th0.68_detr_th0.95.json")
    parser.add_argument("--answer_path", type=str,
                        default="./data/marine_qa/answer")
    parser.add_argument("--answers_file", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_strength", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sampling", action="store_true")

    # ---- Intervention method selection ----
    parser.add_argument(
        "--method", type=str, default="marine",
        choices=["marine", "dynamic_projectaway", "m3id_plus"],
        help="'marine': original CFG at logit level (default).  "
             "'dynamic_projectaway': DPA — layer-specific hidden-state intervention.  "
             "'m3id_plus': M3ID with GammaNet."
    )

    # ---- Dynamic-PROJECTAWAY hyperparameters ----
    parser.add_argument(
        "--hgai_layers", type=str, default="5-18",
        metavar="START-END",
        help="Layer range for HGAI (Visual Information Enrichment). Default: 5-18."
    )
    parser.add_argument(
        "--pa_layers", type=str, default="19-26",
        metavar="START-END",
        help="Layer range for PROJECTAWAY + Sink Redistribution (Semantic Refinement). "
             "Default: 19-26."
    )
    parser.add_argument(
        "--dilution_threshold", type=float, default=0.10,
        help="VAR proxy threshold below which intervention activates (default: 0.10)."
    )
    parser.add_argument(
        "--hgai_amplify_factor", type=float, default=2.0,
        help="Consensus attention amplification factor for HGAI (default: 2.0)."
    )
    parser.add_argument(
        "--sink_threshold", type=float, default=2.0,
        help="An image patch is a sink when its attention > sink_threshold × mean "
             "image attention (default: 2.0)."
    )

    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    eval_model(args)