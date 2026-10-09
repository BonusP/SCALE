import json
import os
from PIL import Image

def get_panel_type(panel_id, panels_dir):
    # panel_id 예: 1-s2.0-S0378775319310869-am_F01-P01
    paper_id = panel_id.split('_F')[0] if '_F' in panel_id else panel_id.split('_T')[0]
    panel_path = os.path.join(panels_dir, paper_id)
    if not os.path.exists(panel_path):
        return "unknown"
    
    for f in os.listdir(panel_path):
        if f.startswith(panel_id) and f.endswith('.png'):
            if '_chart' in f: return 'chart'
            if '_image' in f: return 'image'
            if '_table' in f: return 'table'
    return "unknown"

def update_index_and_crops(index_path, panels_root, output_root):
    with open(index_path, 'r') as f:
        old_index = json.load(f)
    
    new_index = {}
    
    for old_id, info in old_index.items():
        panel_id = info['panel_id']
        # 1-s2.0-S0378775319310869-am_F01-P01 -> F01-P01
        panel_short_id = panel_id.split('_')[-1]
        
        # C## 추출 (기존 __C01 또는 C01)
        suffix = old_id.split('C')[-1] # 01
        
        # 새 ID: F01-P01-C01
        new_comp_short_id = f"{panel_short_id}-C{suffix}"
        # 전체 ID (저장용): paper_id + F01-P01-C01
        paper_id = panel_id.split('_F')[0] if '_F' in panel_id else panel_id.split('_T')[0]
        full_new_id = f"{paper_id}_{new_comp_short_id}"
        
        # 패널 타입 인식
        p_type = get_panel_type(panel_id, panels_root)
        
        # 파일명: full_new_id + _type
        new_filename = f"{full_new_id}_{p_type}.png"
        
        # 인덱스 업데이트
        new_index[full_new_id] = info
        new_index[full_new_id]['filename'] = new_filename
        
        # 크롭 작업
        panel_dir = os.path.join(panels_root, paper_id)
        panel_file = None
        for f in os.listdir(panel_dir):
            if f.startswith(panel_id) and f.endswith('.png'):
                panel_file = f
                break
        
        if panel_file:
            img = Image.open(os.path.join(panel_dir, panel_file))
            bbox = info['bbox_xyxy']
            crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            save_dir = os.path.join(output_root, paper_id)
            os.makedirs(save_dir, exist_ok=True)
            crop_img.save(os.path.join(save_dir, new_filename))
            print(f"Saved: {new_filename}")

    # 인덱스 파일 덮어쓰기
    with open(index_path, 'w') as f:
        json.dump(new_index, f, indent=2)
    print("Updated component_index.json")

if __name__ == "__main__":
    INDEX = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/links/component_index.json"
    PANELS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/panels"
    COMPONENTS = "/mnt/c/Users/HHS/OneDrive/10_Antigravity_Space/01_Projects/PlotLogic/data/datasets/phase1_samples/components"
    
    # 기존 컴포넌트 파일 삭제 (클린 업데이트)
    # import shutil
    # if os.path.exists(COMPONENTS):
    #     shutil.rmtree(COMPONENTS)
    
    update_index_and_crops(INDEX, PANELS, COMPONENTS)
