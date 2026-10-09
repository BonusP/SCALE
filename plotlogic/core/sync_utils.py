import json
import os
import shutil
from PIL import Image

def get_panel_type(panel_id, panels_dir):
    paper_id = panel_id.split('_F')[0] if '_F' in panel_id else panel_id.split('_T')[0]
    panel_path = os.path.join(panels_dir, paper_id)
    if not os.path.exists(panel_path): return "unknown"
    for f in os.listdir(panel_path):
        if f.startswith(panel_id) and f.endswith('.png'):
            if '_chart' in f: return 'chart'
            if '_image' in f: return 'image'
            if '_table' in f: return 'table'
    return "unknown"

def sync_index_and_crops(index_path, panels_root, output_root):
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    paper_id = next(iter(index.keys())).split('_F')[0] if '_F' in next(iter(index.keys())) else next(iter(index.keys())).split('_T')[0]
    save_dir = os.path.join(output_root, paper_id)
    
    # 1. 클린 업데이트를 위해 기존 파일 모두 삭제
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 인덱스 기반 정적 크롭/저장
    for comp_id, info in index.items():
        panel_id = info['panel_id']
        panel_dir = os.path.join(panels_root, paper_id)
        
        # 실제 패널 파일 찾기
        panel_file = None
        for f in os.listdir(panel_dir):
            if f.startswith(panel_id) and f.endswith('.png'):
                panel_file = f
                break
        
        if not panel_file:
            print(f"Warning: Panel {panel_id} not found.")
            continue
            
        img = Image.open(os.path.join(panel_dir, panel_file))
        bbox = info['bbox_xyxy']
        crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 인덱스에 명시된 filename으로 저장
        filename = info['filename']
        crop_img.save(os.path.join(save_dir, filename))
        print(f"Synced & Saved: {filename}")

if __name__ == "__main__":
    INDEX = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/links/component_index.json"
    PANELS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/panels"
    COMPONENTS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/components"
    
    sync_index_and_crops(INDEX, PANELS, COMPONENTS)
