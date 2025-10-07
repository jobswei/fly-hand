import argparse
import os
import trimesh

def convert_glb_to_obj(input_path, output_path, overwrite=False):
    """
    转换单个GLB文件到OBJ
    """
    if not overwrite and os.path.exists(output_path):
        print(f"跳过已存在文件: {output_path}")
        return

    try:
        # 加载GLB模型
        mesh = trimesh.load(input_path)
        
        # 如果是多个网格，合并为一个
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if not meshes:
                print(f"警告: {input_path} 中没有找到网格，已跳过")
                return
            mesh = trimesh.util.concatenate(meshes)
        
        # 导出为OBJ
        mesh.export(output_path, file_type='obj')
        print(f"已转换: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"转换失败: {input_path} -> {e}")

def batch_convert_glb_to_obj(input_dir, output_dir=None, overwrite=False, recursive=True):
    """
    批量转换目录下的所有GLB文件
    """
    if not output_dir:
        output_dir = input_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.glb'):
                glb_path = os.path.join(root, filename)
                
                # 构建输出路径
                rel_path = os.path.relpath(root, input_dir)
                out_root = os.path.join(output_dir, rel_path)
                os.makedirs(out_root, exist_ok=True)
                
                base_name = os.path.splitext(filename)[0]
                obj_path = os.path.join(out_root, f"{base_name}.obj")
                
                convert_glb_to_obj(glb_path, obj_path, overwrite)
                
        if not recursive:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将GLB文件转换为OBJ格式")
    parser.add_argument("input_dir", help="包含GLB文件的输入目录")
    parser.add_argument("-o", "--output_dir", help="OBJ文件的输出目录，默认为输入目录")
    parser.add_argument("-f", "--force", action="store_true", help="覆盖已存在的OBJ文件")
    parser.add_argument("--no-recursive", action="store_true", help="不递归处理子目录")
    
    args = parser.parse_args()
    
    batch_convert_glb_to_obj(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.force,
        recursive=not args.no_recursive
    )