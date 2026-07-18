# 72 SFT + ZS Eval 实验指令

## 权重加载验证

| Model | 状态 | 说明 |
|:---|:---:|:---|
| NaVILA | ✅ 正常 | 所有权重加载成功 |
| StreamVLN | ✅ 正常 | 所有权重加载成功 |
| NaVid | ⚠️ depth fix applied, needs GPU test |

## Eval 输出路径隔离

每个实验用独立的 `checkpoint_folder` 和 `tensorboard_dir`，ZS 和 SFT 互不干扰：

```
evaluation-vln-dpedpro2/{model}_{version}_{type}_{mode}_{split}/
  hm3d/checkpoints/eval_resume_{split}.json
  hm3d/tb/
```

---

## NAVILLA ✅ READY

### navilla v1

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 1 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_zs...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_dynamic_val_unseen.bash` |
| 2 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_zs_d...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_dynamic_val_seen.bash` |
| 3 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_z...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_dynamic_test_unseen.bash` |
| 4 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navill...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_static_val_unseen.bash` |
| 5 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_static_val_seen.bash` |
| 6 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navil...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_zs_static_test_unseen.bash` |
| 7 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_sf...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_dynamic_val_unseen.bash` |
| 8 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_sft_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_dynamic_val_seen.bash` |
| 9 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v1_s...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_dynamic_test_unseen.bash` |
| 10 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navill...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_static_val_unseen.bash` |
| 11 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_static_val_seen.bash` |
| 12 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navil...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v1_sft_static_test_unseen.bash` |

### navilla v2

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 13 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_zs...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_dynamic_val_unseen.bash` |
| 14 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_zs_d...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_dynamic_val_seen.bash` |
| 15 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_z...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_dynamic_test_unseen.bash` |
| 16 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navill...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_static_val_unseen.bash` |
| 17 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_static_val_seen.bash` |
| 18 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navil...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_zs_static_test_unseen.bash` |
| 19 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_sf...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_dynamic_val_unseen.bash` |
| 20 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_sft_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_dynamic_val_seen.bash` |
| 21 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_v2_s...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_dynamic_test_unseen.bash` |
| 22 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navill...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_static_val_unseen.bash` |
| 23 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navilla_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_static_val_seen.bash` |
| 24 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navil...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navilla_v2_sft_static_test_unseen.bash` |

---

## STREAMVLN ✅ READY

### streamvln v1

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 25 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_dynamic_val_unseen.bash` |
| 26 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v1_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_dynamic_val_seen.bash` |
| 27 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_dynamic_test_unseen.bash` |
| 28 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stre...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_static_val_unseen.bash` |
| 29 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stream...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_static_val_seen.bash` |
| 30 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/str...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_zs_static_test_unseen.bash` |
| 31 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_dynamic_val_unseen.bash` |
| 32 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v1_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_dynamic_val_seen.bash` |
| 33 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_dynamic_test_unseen.bash` |
| 34 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stre...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_static_val_unseen.bash` |
| 35 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stream...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_static_val_seen.bash` |
| 36 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/str...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v1_sft_static_test_unseen.bash` |

### streamvln v2

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 37 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_dynamic_val_unseen.bash` |
| 38 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v2_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_dynamic_val_seen.bash` |
| 39 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_dynamic_test_unseen.bash` |
| 40 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stre...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_static_val_unseen.bash` |
| 41 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stream...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_static_val_seen.bash` |
| 42 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/str...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_zs_static_test_unseen.bash` |
| 43 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_dynamic_val_unseen.bash` |
| 44 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_v2_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_dynamic_val_seen.bash` |
| 45 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/streamvln_...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_dynamic_test_unseen.bash` |
| 46 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stre...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_static_val_unseen.bash` |
| 47 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/stream...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_static_val_seen.bash` |
| 48 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/str...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/streamvln_v2_sft_static_test_unseen.bash` |

---

## NAVID ⚠️ depth fix applied (39→40 blocks), pending GPU verification

### navid v1

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 49 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_zs_dyn...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_dynamic_val_unseen.bash` |
| 50 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_zs_dynam...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_dynamic_val_seen.bash` |
| 51 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_zs_dy...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_dynamic_test_unseen.bash` |
| 52 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_static_val_unseen.bash` |
| 53 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_z...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_static_val_seen.bash` |
| 54 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_zs_static_test_unseen.bash` |
| 55 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_sft_dy...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_dynamic_val_unseen.bash` |
| 56 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_sft_dyna...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_dynamic_val_seen.bash` |
| 57 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_sft_d...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_dynamic_test_unseen.bash` |
| 58 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_static_val_unseen.bash` |
| 59 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v1_s...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_static_val_seen.bash` |
| 60 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v1_sft_static_test_unseen.bash` |

### navid v2

| # | Type | Mode | Split | Python | sbatch |
|:---|:---|:---|:---|:---|:---|
| 61 | ZS | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_zs_dyn...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_dynamic_val_unseen.bash` |
| 62 | ZS | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_zs_dynam...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_dynamic_val_seen.bash` |
| 63 | ZS | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_zs_dy...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_dynamic_test_unseen.bash` |
| 64 | ZS | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_static_val_unseen.bash` |
| 65 | ZS | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_z...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_static_val_seen.bash` |
| 66 | ZS | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_zs_static_test_unseen.bash` |
| 67 | SFT | 动态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_sft_dy...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_dynamic_val_unseen.bash` |
| 68 | SFT | 动态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_sft_dyna...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_dynamic_val_seen.bash` |
| 69 | SFT | 动态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_sft_d...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_dynamic_test_unseen.bash` |
| 70 | SFT | 静态行人 | val_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_val_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_static_val_unseen.bash` |
| 71 | SFT | 静态行人 | val_seen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_val_seen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v2_s...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_static_val_seen.bash` |
| 72 | SFT | 静态行人 | test_unseen | `python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v2_test_unseen.yaml habitat_baselines.evaluate=True habitat_baselines.checkpoint_folder=evaluation-vln-dpedpro2/navid_v...` | `sbatch /share/home/u19666033/dhj/dped-vln/sbatch/DPed_vlm/all_eval/navid_v2_sft_static_test_unseen.bash` |

---

**总计: 72 个实验**

- NaVILA: 24, StreamVLN: 24 (48 ready), NaVid: 24 (depth fix applied, pending verification)