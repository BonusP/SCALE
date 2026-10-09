import json
import os
from PIL import Image

def crop_components(index_path, panels_root, output_root):
    with open(index_path, 'r') as f:
        index = json.load(f)

    for comp_id, info in index.items():
        panel_id = info['panel_id']
        # 1-s2.0-S0378775319310869-am_F01-P01 -> 1-s2.0-S0378775319310869-am (folder name)
        paper_id = panel_id.split('_F')[0] if '_F' in panel_id else panel_id.split('_T')[0]
        
        # 실제 파일명 찾기 (Suffix 포함)
        panel_dir = os.path.join(panels_root, paper_id)
        panel_file = None
        for f in os.listdir(panel_dir):
            if f.startswith(panel_id) and f.endswith('.png'):
                panel_file = f
                break
        
        if not panel_file:
            print(f"Warning: Panel file for {panel_id} not found.")
            continue

        img_path = os.path.join(panel_dir, panel_file)
        img = Image.open(img_path)
        
        bbox = info['bbox_xyxy'] # [x1, y1, x2, y2]
        crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 결과 저장 경로
        save_dir = os.path.join(output_root, paper_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{comp_id}.png")
        
        crop_img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    INDEX = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/links/component_index.json"
    PANELS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/panels"
    COMPONENTS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/components"
    
    crop_components(INDEX, PANELS, COMPONENTS)
