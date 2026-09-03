#!/usr/bin/env python3
"""Run the OFFICIAL sam-3d-body pipeline on issue15.jpg (forced bbox, default
cam_int = C++ defaults) and dump every hand-relevant intermediate + the final
spliced result, for numerical comparison against the C++ port.

Run with the official repo's venv:
  /home/ammar/Documents/3dParty/sam-3d-body/venv/bin/python tools/dump_official_hand_gt.py

Outputs (into --out dir, default /tmp/official_hand_gt):
  final_*.npy       final spliced pose_output["mhr"] fields (per person)
  pass1_*.npy       pass-1 body-decoder output (before pass-2 overwrite)
  lhand_raw_*.npy   LEFT  crop's mhr_hand output BEFORE the run_inference
                    "flip hand pose" overwrite (hand[:, :54] still original)
  lhand_*.npy       LEFT  crop's mhr_hand output AFTER  the overwrite
  rhand_*.npy       RIGHT crop's mhr_hand output
  batch_*.npy       hand-crop batch geometry (bbox_center/bbox_scale/affine/img_size)
  gate_*.npy        recomputed 4-criteria hand_valid_mask + intermediates
  summary.txt       human-readable overview of the key hand values
"""
import os
import sys

import numpy as np
import torch

OFFICIAL = "/home/ammar/Documents/3dParty/sam-3d-body"
sys.path.insert(0, OFFICIAL)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402
from sam_3d_body.models.modules.mhr_utils import (  # noqa: E402
    rotation_angle_difference,
    fix_wrist_euler,
)

import roma  # noqa: E402

IMG = "/home/ammar/Documents/Programming/SAM3DBody-cpp/issue15.jpg"
BOX = np.array([[99.33459, 226.46288, 1047.5374, 1373.7561]], dtype=np.float32)

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/official_hand_gt"
os.makedirs(out_dir, exist_ok=True)

def save(name, arr):
    arr = np.asarray(arr)
    np.save(os.path.join(out_dir, name + ".npy"), arr)
    print(f"saved {name}: {arr.shape} {arr.dtype}")

def save_txt(name, text):
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(text)

model, model_cfg = load_sam_3d_body(
    checkpoint_path=OFFICIAL + "/checkpoints/sam-3d-body-dinov3/model.ckpt",
    device="cuda",
    mhr_path=OFFICIAL + "/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
)
estimator = SAM3DBodyEstimator(model, model_cfg)

# ── capture hooks ────────────────────────────────────────────────────────────
cap = {}
hand_step_order = []

def _to_numpy_dict(d):
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out

_orig_forward_step = model.forward_step
def hook_forward_step(batch, decoder_type="body"):
    out = _orig_forward_step(batch, decoder_type)
    if decoder_type == "body":
        cap["pass1_mhr"] = _to_numpy_dict(out["mhr"])
    else:
        hand_step_order.append(decoder_type)
        key = "hand_raw" if len(hand_step_order) == 1 else "hand_raw_r"
        cap[key] = _to_numpy_dict(out["mhr_hand"])
    return out
model.forward_step = hook_forward_step

_orig_run_inference = model.run_inference
def hook_run_inference(img, batch, inference_type="full", transform_hand=None,
                       thresh_wrist_angle=1.4):
    ret = _orig_run_inference(
        img, batch, inference_type=inference_type,
        transform_hand=transform_hand, thresh_wrist_angle=thresh_wrist_angle)
    if len(ret) == 5:
        pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output = ret
        cap["lhand"] = _to_numpy_dict(lhand_output["mhr_hand"])
        cap["rhand"] = _to_numpy_dict(rhand_output["mhr_hand"])
        keep = ("bbox_center", "bbox_scale", "affine_trans", "img_size",
                "ori_img_size")
        cap["batch_lhand"] = _to_numpy_dict(
            {k: batch_lhand[k] for k in keep if k in batch_lhand})
        cap["batch_rhand"] = _to_numpy_dict(
            {k: batch_rhand[k] for k in keep if k in batch_rhand})
        cap["lhand_img"] = batch_lhand["img"].detach().cpu().numpy()
        cap["rhand_img"] = batch_rhand["img"].detach().cpu().numpy()
    return ret
model.run_inference = hook_run_inference

_orig_run_keypoint_prompt = model.run_keypoint_prompt
def hook_run_keypoint_prompt(batch, output, keypoint_prompt):
    out, kp = _orig_run_keypoint_prompt(batch, output, keypoint_prompt)
    # out["mhr"] is now the pass-2 (keypoint-prompted) output, pre-splice:
    # joint_global_rots here is what the "Doing IK" zero_rot_R is built from.
    cap["pass2_mhr"] = _to_numpy_dict(out["mhr"])
    return out, kp
model.run_keypoint_prompt = hook_run_keypoint_prompt

# ── run ──────────────────────────────────────────────────────────────────────
H, W = 1382, 1080
f = float(np.sqrt(W * W + H * H))
cam_int = torch.tensor([[[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]]])

import cv2
img_rgb = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)

outputs = estimator.process_one_image(
    img_rgb, bboxes=BOX, cam_int=cam_int, inference_type="full")
out = outputs[0]

# ── save everything ──────────────────────────────────────────────────────────
for k, v in out.items():
    if isinstance(v, np.ndarray):
        save("final_" + k, v)
for k, v in cap["pass1_mhr"].items():
    save("pass1_" + k, v)
for k, v in cap["lhand"].items():
    save("lhand_" + k, v)
for k, v in cap["rhand"].items():
    save("rhand_" + k, v)
for k, v in cap["hand_raw"].items():
    save("lhand_raw_" + k, v)
for k, v in cap["hand_raw_r"].items():
    save("rhand_raw_" + k, v)
for k, v in cap["batch_lhand"].items():
    save("batch_lhand_" + k, v)
for k, v in cap["batch_rhand"].items():
    save("batch_rhand_" + k, v)
if "pass2_mhr" in cap:
    for k, v in cap["pass2_mhr"].items():
        save("pass2_" + k, v)
if "lhand_img" in cap:
    save("lhand_img", cap["lhand_img"])
    save("rhand_img", cap["rhand_img"])

# ── recompute the gate + splice intermediates (same formulas as run_inference)
body_pose_p1 = torch.from_numpy(cap["pass1_mhr"]["body_pose"]).float()
jgr_p1 = torch.from_numpy(cap["pass1_mhr"]["joint_global_rots"]).float()
ori_local_wrist_rotmat = roma.euler_to_rotmat(
    "XZY", body_pose_p1[:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3)))
save("gate_ori_local_wrist_rotmat", ori_local_wrist_rotmat.numpy())
ori_local_euler = roma.rotmat_to_euler("XZY", ori_local_wrist_rotmat)
save("gate_ori_local_euler", ori_local_euler.numpy())

lowarm = jgr_p1[:, torch.LongTensor([76, 40])]
pre = model.head_pose.joint_rotation[torch.LongTensor([77, 41])].cpu()
wrist_zero_p1 = lowarm @ pre
save("gate_wrist_zero_p1", wrist_zero_p1.numpy())

lgr = torch.from_numpy(cap["lhand"]["joint_global_rots"]).float()
rgr = torch.from_numpy(cap["rhand"]["joint_global_rots"]).float()
pred_global_wrist = torch.stack([lgr[:, 78], rgr[:, 42]], dim=1)
save("gate_pred_global_wrist", pred_global_wrist.numpy())
fused = torch.einsum("kabc,kabd->kadc", pred_global_wrist, wrist_zero_p1)
save("gate_fused_local_wrist", fused.numpy())
angle_diff = rotation_angle_difference(ori_local_wrist_rotmat, fused)
save("gate_angle_diff", angle_diff.numpy())

box_ok = torch.stack([
    (torch.from_numpy(cap["batch_lhand"]["bbox_scale"]).float().flatten(0, 1) > 64).all(dim=1),
    (torch.from_numpy(cap["batch_rhand"]["bbox_scale"]).float().flatten(0, 1) > 64).all(dim=1),
], dim=1)
save("gate_box_ok", box_ok.numpy())

kps_ok = torch.stack([
    torch.from_numpy(cap["lhand"]["pred_keypoints_2d_cropped"]).float().abs().amax(dim=(1, 2)) < 0.5,
    torch.from_numpy(cap["rhand"]["pred_keypoints_2d_cropped"]).float().abs().amax(dim=(1, 2)) < 0.5,
], dim=1)
save("gate_kps_ok", kps_ok.numpy())

kps_right_wrist_idx, kps_left_wrist_idx = 41, 62
right_kps_full = torch.from_numpy(cap["rhand"]["pred_keypoints_2d"]).float()[:, [41]].clone()
left_kps_full = torch.from_numpy(cap["lhand"]["pred_keypoints_2d"]).float()[:, [41]].clone()
left_kps_full[:, :, 0] = W - left_kps_full[:, :, 0] - 1
body_kps = torch.from_numpy(cap["pass1_mhr"]["pred_keypoints_2d"]).float()
body_right = body_kps[:, [41]].clone()
body_left = body_kps[:, [62]].clone()
right_dist = (right_kps_full - body_right).flatten(0, 1).norm(dim=-1) / \
    torch.from_numpy(cap["batch_lhand"]["bbox_scale"]).float().flatten(0, 1)[:, 0]
left_dist = (left_kps_full - body_left).flatten(0, 1).norm(dim=-1) / \
    torch.from_numpy(cap["batch_rhand"]["bbox_scale"]).float().flatten(0, 1)[:, 0]
save("gate_right_wrist_dist", right_dist.numpy())
save("gate_left_wrist_dist", left_dist.numpy())
wrist_ok = torch.stack([left_dist < 0.25, right_dist < 0.25], dim=1)
save("gate_wrist_ok", wrist_ok.numpy())

hand_valid = (angle_diff < 1.4) & box_ok & kps_ok & wrist_ok
save("gate_hand_valid_mask", hand_valid.numpy())

# ── pass-2 (Doing IK) splice internals ───────────────────────────────────────
if "pass2_mhr" in cap:
    jgr_p2 = torch.from_numpy(cap["pass2_mhr"]["joint_global_rots"]).float()
    lowarm_p2 = jgr_p2[:, torch.LongTensor([76, 40])]
    wrist_zero_p2 = lowarm_p2 @ pre
    save("splice_wrist_zero_p2", wrist_zero_p2.numpy())
    fused_p2 = torch.einsum("kabc,kabd->kadc", pred_global_wrist, wrist_zero_p2)
    save("splice_fused_p2", fused_p2.numpy())
    wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_p2))
    save("splice_wrist_xzy", wrist_xzy.numpy())
    valid_angle = (rotation_angle_difference(ori_local_wrist_rotmat, fused_p2)
                   < 1.4) & hand_valid
    save("splice_valid_angle", valid_angle.numpy())
    splice_summary = [
        f"splice_valid_angle = {valid_angle.numpy()}",
        f"official spliced wrist_xzy (x,z,y): {wrist_xzy.numpy()}",
    ]
else:
    splice_summary = []

# ── human-readable summary ───────────────────────────────────────────────────
lines = []
lines.append("== gate ==")
lines.append(f"angle_diff       = {angle_diff.numpy()}")
lines.append(f"box_ok           = {box_ok.numpy()}")
lines.append(f"kps3_ok          = {kps_ok.numpy()}")
lines.append(f"wrist dist l/r   = {left_dist.item():.4f} / {right_dist.item():.4f}")
lines.append(f"wrist_ok         = {wrist_ok.numpy()}")
lines.append(f"hand_valid_mask  = {hand_valid.numpy()}")
lines += splice_summary
lines.append("")
lines.append("== final spliced ==")
for k in ("global_rot", "body_pose_params", "hand_pose_params", "scale_params",
          "shape_params", "pred_cam_t"):
    v = out[k]
    lines.append(f"{k}: {np.array2string(np.asarray(v), precision=4, suppress_small=False)}")
lines.append("")
lines.append("== lhand crop (post flip-hand-pose overwrite) ==")
lines.append(f"hand[:54]  = {np.array2string(cap['lhand']['hand'][0, :54], precision=4)}")
lines.append(f"hand[54:]  = {np.array2string(cap['lhand']['hand'][0, 54:], precision=4)}")
lines.append("== lhand crop RAW (pre overwrite) ==")
lines.append(f"hand[:54]  = {np.array2string(cap['hand_raw']['hand'][0, :54], precision=4)}")
lines.append(f"hand[54:]  = {np.array2string(cap['hand_raw']['hand'][0, 54:], precision=4)}")
lines.append("== rhand crop ==")
lines.append(f"hand[:54]  = {np.array2string(cap['rhand']['hand'][0, :54], precision=4)}")
lines.append(f"hand[54:]  = {np.array2string(cap['rhand']['hand'][0, 54:], precision=4)}")
lines.append("")
lines.append("== scale/shape per crop ==")
lines.append(f"lhand scale = {np.array2string(cap['lhand']['scale'][0], precision=4)}")
lines.append(f"rhand scale = {np.array2string(cap['rhand']['scale'][0], precision=4)}")
lines.append(f"final scale = {np.array2string(out['scale_params'], precision=4)}")
lines.append(f"lhand shape = {np.array2string(cap['lhand']['shape'][0], precision=4)}")
lines.append(f"rhand shape = {np.array2string(cap['rhand']['shape'][0], precision=4)}")
lines.append(f"final shape = {np.array2string(out['shape_params'], precision=4)}")
lines.append("")
lines.append(f"lhand bbox_scale = {np.array2string(cap['batch_lhand']['bbox_scale'], precision=3)}")
lines.append(f"rhand bbox_scale = {np.array2string(cap['batch_rhand']['bbox_scale'], precision=3)}")
lines.append(f"lhand bbox_center= {np.array2string(cap['batch_lhand']['bbox_center'], precision=3)}")
lines.append(f"rhand bbox_center= {np.array2string(cap['batch_rhand']['bbox_center'], precision=3)}")
lines.append(f"lhand affine     = {np.array2string(cap['batch_lhand']['affine_trans'][0], precision=4)}")
lines.append(f"rhand affine     = {np.array2string(cap['batch_rhand']['affine_trans'][0], precision=4)}")
lines.append("")
lines.append(f"lhand pred_cam_t = {np.array2string(cap['lhand']['pred_cam_t'][0], precision=4)}")
lines.append(f"rhand pred_cam_t = {np.array2string(cap['rhand']['pred_cam_t'][0], precision=4)}")
lines.append(f"final pred_cam_t = {np.array2string(out['pred_cam_t'], precision=4)}")
lines.append("")
lines.append("== lhand/rhand wrist 2D (kp 41) full-image ==")
lines.append(f"lhand kp41 = {cap['lhand']['pred_keypoints_2d'][0, 41]} (x is in FLIPPED frame)")
lines.append(f"rhand kp41 = {cap['rhand']['pred_keypoints_2d'][0, 41]}")
lines.append(f"final kp41 = {out['pred_keypoints_2d'][41]}")
lines.append(f"final kp62 = {out['pred_keypoints_2d'][62]}")
lines.append("")
# scale derivation constants for the left-hand scale[9] PCA trick
lines.append("== left scale[9] PCA derivation (head_pose.scale_mean/comps) ==")
lines.append(f"scale_mean[8] = {model.head_pose.scale_mean[8].item():.6f}")
lines.append(f"scale_mean[9] = {model.head_pose.scale_mean[9].item():.6f}")
lines.append(f"scale_comps[8,8] = {model.head_pose.scale_comps[8, 8].item():.6f}")
lines.append(f"scale_comps[9,9] = {model.head_pose.scale_comps[9, 9].item():.6f}")
lines.append(f"lhand scale[8] raw = {cap['hand_raw']['scale'][0, 8]:.6f}")
lines.append(f"lhand scale[9] after derivation = {cap['lhand']['scale'][0, 9]:.6f}")
save_txt("summary.txt", "\n".join(lines))
print("\n".join(lines))
print("\ndone")
