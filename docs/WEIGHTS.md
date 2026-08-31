# Identity adapter weights

The LoRA adapter that supplies `gender` and `age`. The momentary attributes
(`exposed`, `watched`) run on the base model and need none of this.

| | |
|---|---|
| download | **&lt;PASTE SHARE LINK HERE&gt;** |
| archive | `identity_lora.tar.gz`, 0.94 GB |
| unpacked | 1.2 GB |

## Install

```bash
pip install gdown
bash setup/fetch_weights.sh --gdrive <SHARE_URL>
```

Or, from a directory you already have:

```bash
bash setup/fetch_weights.sh /path/to/identity_lora
```

## Verify

`fetch_weights.sh` prints these after installing. They should match exactly.

| file | size | sha256 |
|---|---|---|
| `adapter_model.safetensors` | 330.3 MB | `63dc9e9ec6df2b4ee100f84e3c5cdcbaef21952efc65d649c4e526cefb8af5c0` |
| `non_lora_state_dict.bin` | 869.9 MB | `408e9eeb88df3985532cac345cd4970e0f700e785ba140c9df7cde1eec562ad3` |

Archive: `336a52f8811fe9d4b16a9f14b49e67f6668f5230cc2c08506a83ffea6e2657fd`

## Why both files matter

`PeftModel.from_pretrained` loads `adapter_model.safetensors` and **silently
ignores** `non_lora_state_dict.bin`, which holds the trained vision tower and
merger. Nothing raises. A copy missing the second file evaluates a base vision
tower underneath a fine-tuned adapter — a model that never existed — and it
scores plausibly enough to look correct.

`setup/check_env.py` refuses to run if that file is absent, and `pipeline/`
loads it explicitly rather than through PEFT alone.

## What is not in the archive

Optimiser state, scheduler state and the mid-training checkpoint. Those are
~9.4 GB and serve only to resume training; the adapter inside that checkpoint is
byte-identical to the one shipped here. Excluding them keeps the download at
0.94 GB and removes the chance of pointing `IDENTITY_ADAPTER` at the wrong
directory.
