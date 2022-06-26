import shutil
import os

parent_dir = {
    'CASMEII':"CASMEII/CASME2_RAW_selected_cropped",
    'SAMM':"SAMM/SAMM_cropped",
    'SMIC':"SMIC/SMIC_all_raw/HS_cropped"
}

if __name__=='__main__':
    os.makedirs("./Mixed_dataset/train", exist_ok=True)
    for dataset in ["SAMM","SMIC"]:
        for ID in os.listdir(os.path.join(parent_dir[dataset])):
            for item in os.listdir(os.path.join(parent_dir[dataset],ID)):
                src_folder = os.path.join(os.path.join(parent_dir[dataset],ID,item))
                # target_folder = os.path.join("/data/home-ustc/xgc18/competition/MEGC2022/Code/Facial-Prior-Based-FOMM-main/data/Mixed_dataset/train",dataset+'_'+item)
                target_folder = os.path.join("./Mixed_dataset/train",dataset+'_'+item)
                shutil.copytree(src_folder,target_folder)

    for dataset in ["CASMEII"]:
        for ID in os.listdir(os.path.join(parent_dir[dataset])):
            for item in os.listdir(os.path.join(parent_dir[dataset],ID)):
                src_folder = os.path.join(os.path.join(parent_dir[dataset],ID,item))
                # target_folder = os.path.join("/data/home-ustc/xgc18/competition/MEGC2022/Code/Facial-Prior-Based-FOMM-main/data/Mixed_dataset/train",dataset+'_'+ID+'_'+item)
                target_folder = os.path.join("./Mixed_dataset/train",dataset+'_'+ID+'_'+item)
                shutil.copytree(src_folder,target_folder)
    
    os.makedirs("./Mixed_dataset/test", exist_ok=True)
    for fold in os.listdir(os.path.join("megc2022-synthesis/source_samples_cropped")):
        for item in os.listdir(os.path.join("megc2022-synthesis/source_samples_cropped", fold)):
            src_fold = os.path.join("megc2022-synthesis/source_samples_cropped", fold, item)
            tgt_fold = os.path.join("./Mixed_dataset/test", item)
            shutil.copytree(src_fold, tgt_fold)
    
    name2dir={
        "Template_Female_Asian.jpg": "normalized_asianFemale",
        "Template_Female_Europe.jpg": "normalized_westernFemale",
        "Template_Male_Asian.jpg": "normalized_asianMale",
        "Template_Male_Europe.JPG": "normalized_westernMale"
    }
    
    for name in os.listdir(os.path.join("megc2022-synthesis/target_template_face_cropped")):
        os.makedirs(os.path.join("./Mixed_dataset/test",name2dir[name]), exist_ok=True)
        src_img = os.path.join("megc2022-synthesis/target_template_face_cropped", name)
        tgt_img = os.path.join("./Mixed_dataset/test",name2dir[name], name)
        shutil.copy(src_img, tgt_img)

    