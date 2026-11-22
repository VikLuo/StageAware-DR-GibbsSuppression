#!/usr/bin/env python3
"""Reconstructed inference pipeline for DR pathology-stage-aware enhancement.

Usage is documented in README.md. This software does not recreate lost
historical predictions or validate previously reported numerical results.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from src.models import build_convnext_small, load_checkpoint
from src.preprocessing import list_original_images, paired_enhancement, load_tensor


def parse_args():
    p=argparse.ArgumentParser(description="DR selective-enhancement 5-class inference")
    p.add_argument("--original-dir", required=True, type=Path)
    p.add_argument("--enhanced-dir", type=Path, help="Directory with *_enhanced.png (required for two_stage/all_enhanced)")
    p.add_argument("--stage2-weights", required=True, type=Path, help="5-class grading checkpoint")
    p.add_argument("--stage1-weights", type=Path, help="2-class routing checkpoint for two_stage")
    p.add_argument("--mode", choices=["two_stage", "raw", "all_enhanced"], default="two_stage")
    p.add_argument("--enhance-class", choices=[0,1], type=int, default=1,
                   help="Gate output index meaning enhance. Confirm mapping against training logs (default 1 is a reconstruction assumption).")
    p.add_argument("--stage-boundary", choices=["grade01", "grade012"], default="grade01",
                   help="Metadata documenting the trained gate's intended boundary; must match its checkpoint.")
    p.add_argument("--head-style", choices=["legacy", "linear"], default="legacy")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, default=Path("predictions.csv"))
    p.add_argument("--fallback-original", action="store_true",
                   help="Use original if a required enhanced image is absent; flagged in output")
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (spec=="cuda" or (spec=="auto" and torch.cuda.is_available())) else "cpu")


def run():
    a=parse_args()
    if a.mode == "two_stage" and not a.stage1_weights:
        raise ValueError("two_stage requires --stage1-weights")
    if a.mode != "raw" and not a.enhanced_dir:
        raise ValueError("Enhancement mode requires --enhanced-dir")
    device=resolve_device(a.device)
    grader=load_checkpoint(build_convnext_small(5,a.head_style,a.dropout),a.stage2_weights,device)
    gate=None
    if a.mode == "two_stage":
        gate=load_checkpoint(build_convnext_small(2,a.head_style,a.dropout),a.stage1_weights,device)
    inputs=list_original_images(a.original_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fieldnames=["image_id","mode","stage_boundary","stage1_class","enhance_class","enhance_selected",
                "source_used","fallback_used","predicted_grade","confidence"]+[f"prob_grade_{i}" for i in range(5)]
    results=[]
    with torch.inference_mode():
        for i,original in enumerate(inputs,1):
            gate_class=""
            choose_enhancement=(a.mode=="all_enhanced")
            if gate is not None:
                gate_class=int(gate(load_tensor(original).to(device)).argmax(dim=1).item())
                choose_enhancement=(gate_class==a.enhance_class)
            actual=original
            fallback=False
            if choose_enhancement:
                try:
                    actual=paired_enhancement(original,a.enhanced_dir)
                except FileNotFoundError:
                    if not a.fallback_original:
                        raise
                    fallback=True
                    actual=original
            logits=grader(load_tensor(actual).to(device))
            probabilities=torch.softmax(logits,dim=1).squeeze(0).cpu().tolist()
            predicted=int(max(range(5),key=lambda k:probabilities[k]))
            record={"image_id":original.name,"mode":a.mode,"stage_boundary":a.stage_boundary,
                    "stage1_class":gate_class,"enhance_class":a.enhance_class if gate is not None else "",
                    "enhance_selected":int(choose_enhancement),"source_used":actual.name,
                    "fallback_used":int(fallback),"predicted_grade":predicted,
                    "confidence":round(probabilities[predicted],8),
                    **{f"prob_grade_{k}":round(probabilities[k],8) for k in range(5)}}
            results.append(record)
            if i%100==0:
                print(f"Processed {i}/{len(inputs)}",flush=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader();writer.writerows(results)
    metadata={"images_processed":len(inputs),"mode":a.mode,"stage_boundary":a.stage_boundary,
              "enhance_class":a.enhance_class,"device":str(device),
              "stage1_checkpoint":str(a.stage1_weights) if a.stage1_weights else None,
              "stage2_checkpoint":str(a.stage2_weights),
              "description":"Reconstructed inference implementation; not validation of lost experimental OOF metrics."}
    a.output.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Saved",len(results),"predictions to",a.output)


if __name__=="__main__":
    try:
        run()
    except Exception as e:
        print("Inference failed:",e,file=sys.stderr)
        raise
