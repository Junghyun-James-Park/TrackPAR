import json
import os
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from trainer import QwenSFTTrainer
from dataset import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib


def _compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _match_persons(pred_list, gt_list, iou_threshold=0.3):
    """Greedy IoU matching: match each GT person to the best available prediction."""
    matched = []
    used = set()
    for gt in gt_list:
        gt_bbox = gt.get("bbox")
        if not gt_bbox or len(gt_bbox) != 4:
            continue
        best_iou, best_idx = iou_threshold, -1
        for i, pred in enumerate(pred_list):
            if i in used:
                continue
            pred_bbox = pred.get("bbox")
            if not pred_bbox or len(pred_bbox) != 4:
                continue
            iou = _compute_iou(gt_bbox, pred_bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0:
            matched.append((pred_list[best_idx], gt, best_iou))
            used.add(best_idx)
    return matched


def _parse_pred_str_train(s: str) -> list:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        end = len(lines)-1 if lines[-1].strip() == "```" else len(lines)
        s = "\n".join(lines[1:end]).strip()
    try:
        p = json.loads(s)
    except Exception:
        last = s.rfind("}")
        try:
            p = json.loads(s[:last+1] + "\n]") if last != -1 else []
        except Exception:
            return []
    if not isinstance(p, list):
        p = [p] if isinstance(p, dict) else []
    for x in p:
        if isinstance(x, dict) and "bbox_2d" in x and "bbox" not in x:
            x["bbox"] = x.pop("bbox_2d")
    return p


def _binary_metrics_train(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total       if total       > 0 else 0.0
    prec = tp / (tp + fp)          if (tp + fp)   > 0 else 0.0
    rec  = tp / (tp + fn)          if (tp + fn)   > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec)   if (prec+rec)  > 0 else 0.0
    return acc, prec, rec, f1


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    references  = eval_pred.references

    all_ious        = []
    gender_correct  = []
    age_errors      = []
    age_within_5    = []
    age_within_10   = []
    age_bucket_corr = []
    exp_tp = exp_fp = exp_fn = exp_tn = 0
    wat_tp = wat_fp = wat_fn = wat_tn = 0
    n_gt_total, n_matched_total = 0, 0

    for pred_str, ref_str in zip(predictions, references):
        pred_list = _parse_pred_str_train(pred_str)
        gt_list   = _parse_pred_str_train(ref_str)

        n_gt_total += len(gt_list)
        matched = _match_persons(pred_list, gt_list)
        n_matched_total += len(matched)

        for pred_p, gt_p, iou in matched:
            all_ious.append(iou)

            if "gender" in pred_p and "gender" in gt_p:
                gender_correct.append(int(str(pred_p["gender"]).upper() == str(gt_p["gender"]).upper()))

            if "age" in pred_p and "age" in gt_p:
                try:
                    p_age = int(pred_p["age"]); g_age = int(gt_p["age"])
                    err   = abs(p_age - g_age)
                    age_errors.append(err)
                    age_within_5.append(int(err <= 5))
                    age_within_10.append(int(err <= 10))
                    age_bucket_corr.append(int(p_age // 10 == g_age // 10))
                except Exception:
                    pass

            if "exposed" in pred_p and "exposed" in gt_p:
                p_e = bool(pred_p["exposed"]); g_e = bool(gt_p["exposed"])
                if   p_e and g_e:     exp_tp += 1
                elif p_e and not g_e: exp_fp += 1
                elif not p_e and g_e: exp_fn += 1
                else:                 exp_tn += 1

            if "watched" in pred_p and "watched" in gt_p:
                p_w = bool(pred_p["watched"]); g_w = bool(gt_p["watched"])
                if   p_w and g_w:     wat_tp += 1
                elif p_w and not g_w: wat_fp += 1
                elif not p_w and g_w: wat_fn += 1
                else:                 wat_tn += 1

    exp_acc, exp_prec, exp_rec, exp_f1 = _binary_metrics_train(exp_tp, exp_fp, exp_fn, exp_tn)
    wat_acc, wat_prec, wat_rec, wat_f1 = _binary_metrics_train(wat_tp, wat_fp, wat_fn, wat_tn)

    return {
        "mean_iou":       float(np.mean(all_ious))            if all_ious        else 0.0,
        "recall":         float(n_matched_total / n_gt_total) if n_gt_total      else 0.0,
        "gender_acc":     float(np.mean(gender_correct))      if gender_correct  else 0.0,
        "age_mae":        float(np.mean(age_errors))          if age_errors      else 0.0,
        "age_within_5":   float(np.mean(age_within_5))        if age_within_5    else 0.0,
        "age_within_10":  float(np.mean(age_within_10))       if age_within_10   else 0.0,
        "age_bucket_acc": float(np.mean(age_bucket_corr))     if age_bucket_corr else 0.0,
        "exposed_acc":    exp_acc,
        "exposed_prec":   exp_prec,
        "exposed_rec":    exp_rec,
        "exposed_f1":     exp_f1,
        "watched_acc":    wat_acc,
        "watched_prec":   wat_prec,
        "watched_rec":    wat_rec,
        "watched_f1":     wat_f1,
        "n_matched":      float(n_matched_total),
        "n_gt":           float(n_gt_total),
    }

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = backbone.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = backbone.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(backbone.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = backbone.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
