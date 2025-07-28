import os
from pathlib import Path
import shutil

test_data_path = "../data/train"

def preprocess_dir(dir_path:str|Path):
    for file in os.listdir(dir_path):
        inner_dir_path =  dir_path / file
        if '.dcm' in file:
            if '_' in file:
                print(file)
                exit()
        # if '_' in file:
        #     continue
        if os.path.isdir(inner_dir_path):
            for in_file in os.listdir(inner_dir_path):
                if '.dcm' in in_file:
                    if '_' in in_file:
                        print(in_file)
                        exit()
                    else:
                        continue
                # source_dir = inner_dir_path / in_file
                # for f in os.listdir(source_dir):
                #     if ".dcm" not in f:
                #         print(f"This dir {source_dir} is not only dcm containing: {f}")
                new_dir_path = inner_dir_path

                if '_' not in file:
                    try:
                        new_dir_name=f"{file}_{in_file}"
                        new_dir_path=f"{dir_path}\\{new_dir_name}"
                        os.rename(inner_dir_path,new_dir_path)
                    except OSError as e:
                        print(f"OS Error: {e}")
                        new_dir_path = inner_dir_path
                        print(new_dir_path)
                inner_inner_dir_path = f"{new_dir_path}\\{in_file}"
                if os.path.exists(inner_inner_dir_path):

                    if os.path.isdir(inner_inner_dir_path) and len(os.listdir(inner_inner_dir_path))>0:
                        try:
                            for f in os.listdir(inner_inner_dir_path):
                                shutil.move(f"{inner_inner_dir_path}\\{f}", f"{new_dir_path}\\{f}")
                            print(f"Directory '{inner_inner_dir_path}' moved to '{new_dir_path}'.")
                        except shutil.Error as e:
                            print(f"Error moving or removing directory: {e}")
                        except OSError as e:
                            print(f"OS Error: {e}")
                    if len(os.listdir(inner_inner_dir_path)) > 0:
                        print(f"Should be empty: {inner_inner_dir_path}")
                    else:
                        shutil.rmtree(inner_inner_dir_path)
                        print(f"Original source directory '{inner_inner_dir_path}' removed.")


        else:
            print(f"This dir {file} is not only dir containing")
    test_set = set()
    test_list = []
    for file in os.listdir(dir_path):
        inner_dir_path = dir_path / file
        for in_file in os.listdir(inner_dir_path):
            test_set.add(in_file)
            test_list.append(in_file)
    assert len(test_set) == len(test_list)

preprocess_dir(Path(test_data_path))