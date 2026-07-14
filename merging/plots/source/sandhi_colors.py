# Sandhi Paper Color Scheme
# Based on colorbrewer2.org - colorblind safe/print friendly/photocopy friendly

# colorbrewer2.org palettes:
# blues: ['#deebf7','#9ecae1','#3182bd']
# greens: ['#e5f5e0','#a1d99b','#31a354']
# greys: ['#f0f0f0','#bdbdbd','#636363']
# oranges: ['#fee6ce','#fdae6b','#e6550d']
# purples: ['#efedf5','#bcbddc','#756bb1']
# reds: ['#fee0d2','#fc9272','#de2d26']

systems = {
    "sandhi": {
        "name": "Sandhi",
        "color": '#e41a1c',  # bright red (ColorBrewer Set1)
        "edge": "#000000",
        "hatch": ""
    },
    "vllm": {
        "name": "No-merge",
        "color": '#377eb8',  # blue (ColorBrewer Set1)
        "hatch": "//"
    },
    "fullmerge": {
        "name": "Full-merge",
        "color": '#4daf4a',  # green (ColorBrewer Set1)
        "hatch": "\\\\"
    },
    "lora": {
        "name": "LoRA",
        "color": '#ff7f00',  # orange (ColorBrewer Set1)
        "hatch": "--"
    }
}

# Model families
models = {
    "deepseek": {
        "name": "DeepSeek-7B",
        "baseline_memory_gb": 14.0,  # GB for full-merge/lora
    },
    "qwen": {
        "name": "Qwen-7B",
        "baseline_memory_gb": 14.0,
    },
    "llama": {
        "name": "Llama-3-8B",
        "baseline_memory_gb": 16.0,
    }
}
