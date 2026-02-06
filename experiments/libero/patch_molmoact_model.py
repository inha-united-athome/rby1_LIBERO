#!/usr/bin/env python3
"""
Patch HuggingFace-cached MolmoAct model code for transformers compatibility.

Known issues in the cached modeling_molmoact.py:
  1) ViTMultiHeadDotProductAttention: attn_implementation can be None when
     config._attn_implementation is not set. Should default to "eager".
  2) flash_attention_2 path references self.config.float32_attention but the
     class stores it as self.float32_attention (no self.config attribute).

Run this script before loading MolmoAct models, or use the auto-patch in
run_rby1_molmoact.sh.
"""

import os
import glob
import re


def patch_modeling_file(filepath: str) -> bool:
    """Apply patches to a single modeling_molmoact.py file. Returns True if any changes made."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    
    # Fix 1: Default attn_implementation to "eager" when None
    content = content.replace(
        'self.attn_implementation = attn_implementation\n',
        'self.attn_implementation = attn_implementation or "eager"\n'
    )
    
    # Fix 2: self.config.float32_attention -> self.float32_attention
    content = content.replace(
        'self.config.float32_attention',
        'self.float32_attention'
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/allenai")
    pattern = os.path.join(cache_dir, "MolmoAct-*", "*", "modeling_molmoact.py")
    
    files = glob.glob(pattern)
    if not files:
        print("No MolmoAct model files found in HuggingFace cache.")
        print(f"  Searched: {pattern}")
        print("  The patches will be applied automatically when you first load the model.")
        return
    
    for filepath in files:
        rel = os.path.relpath(filepath, cache_dir)
        if patch_modeling_file(filepath):
            print(f"[patched] {rel}")
        else:
            print(f"[ok]      {rel} (already patched or no changes needed)")


if __name__ == "__main__":
    main()
