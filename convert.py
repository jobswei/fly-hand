import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import trimesh


def create_mujoco_xml(config, output_path):
    """
    根据配置生成带地面支撑力的MuJoCo XML文件
    :param config: 物体配置字典（包含asset、position、rotation、size）
    :param output_path: 输出XML文件路径
    """
    # 1. 创建根元素
    mujoco = ET.Element('mujoco', model='objects_scene')

    # 2. 全局选项设置（启用重力，timestep控制物理模拟步长）
    ET.SubElement(mujoco, 'option', timestep="0.01", gravity="0 0 -9.81")

    # 3. 默认参数配置（统一设置关节和几何体属性，避免重复）
    default = ET.SubElement(mujoco, 'default')
    # 关节配置：确保物体运动平滑（阻尼防止过快摆动）
    ET.SubElement(
        default,
        'joint',
        armature="0.1",  # 关节转动惯量
        damping="1",  # 关节阻尼
        limited="true"  # 关节是否有限位（此处为通用配置，可根据需求调整）
    )
    # 几何体配置：核心参数确保碰撞和支撑力生效
    ET.SubElement(
        default,
        'geom',
        conaffinity="1",  # 碰撞亲和力：1=允许与所有物体碰撞（0=不参与碰撞）
        condim="3",  # 碰撞维度：3=完整3D碰撞（支撑力需竖直方向力反馈）
        friction="1 0.1 0.1",  # 摩擦系数（主摩擦+滚动摩擦+自旋摩擦，防止滑动）
        density="1000",  # 物体密度（kg/m³，用于计算质量，有质量才受重力）
        margin="0.002"  # 碰撞边际（防止物体因精度问题穿透）
    )

    # 4. 资产定义（导入OBJ模型，为每个物体创建独立mesh）
    asset = ET.SubElement(mujoco, 'asset')

    # 5. 世界体（放置所有物体和地面）
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 6. 物体计数器（确保同名物体有唯一ID，避免XML标签重复）
    object_counter = {}

    # 7. 遍历配置，生成每个物体的XML节点
    for obj in config['objects']:
        # 7.1 获取物体基础信息
        asset_name = obj['asset']
        position = obj['position']
        rotation = obj['rotation']
        target_size = obj['size']  # 目标最大维度尺寸（用于自动缩放）

        # 7.2 生成唯一物体ID
        if asset_name not in object_counter:
            object_counter[asset_name] = 1
        else:
            object_counter[asset_name] += 1
        obj_unique_id = f"{asset_name}_{object_counter[asset_name]}"

        # 7.3 物体OBJ模型路径映射（需确保路径与实际文件一致）
        asset_paths = {
            'coffee_cup':
            "./scene_assets/living_room/Coffee_cup_withe_obj/Coffee_cup_withe_.obj",
            'yellow_barrier':
            "./scene_assets/living_room/Cone_Buoy/conbyfr.obj",
            'book':
            "./scene_assets/living_room/Book_by_Peter_Iliev_obj/Book_by_Peter_Iliev_obj.obj",
            'football':
            "./scene_assets/living_room/soccer/Obj.obj",
            'basketball':
            "./scene_assets/living_room/basketball/BasketBall.obj",
            'table':
            "scene_assets/env_assets/CENSI_COFTABLE_obj/Censi_ConcreteCoffeeTable_free_obj.obj"
        }

        # 7.4 跳过路径未定义的物体
        if asset_name not in asset_paths:
            print(f"⚠️  警告：未找到 '{asset_name}' 的模型路径，已跳过该物体")
            continue
        obj_mesh_path = asset_paths[asset_name]

        # 7.5 基于AABB自动计算缩放比例（确保物体最大维度符合target_size）
        try:
            # 加载模型并计算原始AABB尺寸（轴对齐边界框）
            mesh = trimesh.load(obj_mesh_path)
            original_bounds = mesh.bounds  # bounds[0]最小坐标，bounds[1]最大坐标
            original_size = np.abs(original_bounds[1] -
                                   original_bounds[0])  # 原始各轴尺寸
            max_original_size = max(original_size)  # 原始最大维度
            scale_ratio = target_size / max_original_size  # 缩放比例（归一化最大维度）
            print(
                f"✅ {obj_unique_id}：原始最大尺寸={max_original_size:.4f}，缩放比例={scale_ratio:.6f}"
            )
        except Exception as e:
            # 模型加载失败时使用默认缩放（避免程序崩溃）
            scale_ratio = 0.01
            print(f"❌ {obj_unique_id}：模型加载失败（{str(e)}），使用默认缩放比例={scale_ratio}")

        # 7.6 在asset中定义该物体的独立mesh（带缩放）
        ET.SubElement(
            asset,
            'mesh',
            name=f"{obj_unique_id}_mesh",
            file=obj_mesh_path,
            scale=f"{scale_ratio} {scale_ratio} {scale_ratio}"  # 三轴等比例缩放
        )

        # 7.7 创建物体的body节点（位置+旋转）
        obj_body = ET.SubElement(
            worldbody,
            'body',
            name=obj_unique_id,
            pos=f"{position[0]} {position[1]} {position[2]}",  # 物体初始位置
            euler=f"{rotation[0]} {rotation[1]} {rotation[2]}"  # 物体初始旋转（欧拉角）
        )

        # 7.8 添加自由关节（允许物体6自由度运动：3平移+3旋转）
        ET.SubElement(obj_body, 'freejoint')

        # 7.9 添加物体的几何体（关联前面定义的mesh）
        ET.SubElement(
            obj_body,
            'geom',
            name=f"{obj_unique_id}_geom",
            type="mesh",
            mesh=f"{obj_unique_id}_mesh"  # 关联asset中的mesh
        )

    # 8. 添加地面（提供支撑力，物体下落接触后停止）
    ET.SubElement(
        worldbody,
        'geom',
        name="floor",
        type="plane",
        size="5 5 0.1",  # 地面尺寸（x=5, y=5, 厚度=0.1，覆盖场景范围）
        rgba="0.9 0.9 0.9 1",  # 地面颜色（RGB+透明度，浅灰色）
        condim="3",  # 地面碰撞维度（必须3D才能提供竖直支撑力）
        density="0",  # 地面密度=0（质量无穷大，不会被物体推动）
        friction="1 0.1 0.1"  # 地面摩擦（与物体摩擦系数一致，防止滑动）
    )

    # 9. 添加执行器节点（空节点，后续如需添加控制器可扩展）
    ET.SubElement(mujoco, 'actuator')

    # 10. 格式化XML（增加缩进，便于阅读）
    rough_xml = ET.tostring(mujoco, 'utf-8')
    parsed_xml = minidom.parseString(rough_xml)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")

    # 11. 去除多余空行（minidom会生成多余空行，优化输出）
    clean_xml = '\n'.join(
        [line for line in pretty_xml.split('\n') if line.strip()])

    # 12. 写入XML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_xml)

    print(f"\n🎉 MuJoCo XML文件已生成：{output_path}")


# -------------------------- 示例配置数据 --------------------------
if __name__ == "__main__":
    # 物体配置字典（可根据需求修改position、rotation、size等参数）
    object_config = {
        "objects": [{
            "asset":
            "coffee_cup",
            "position":
            [0.6175566804667381, 0.4018943018251386, 0.31937398627448127],
            "rotation": [90, 0, 0],
            "size":
            0.098
        }, {
            "asset":
            "yellow_barrier",
            "position":
            [0.05130502865257103, 0.6606247730319297, 0.31637398627448127],
            "rotation": [90, 0, 0],
            "size":
            0.092
        }, {
            "asset":
            "book",
            "position":
            [0.06635277244501003, 0.473959091311956, 0.31437398627448127],
            "rotation": [90, 0, 0],
            "size":
            0.088
        }, {
            "asset":
            "football",
            "position":
            [-0.3885905482642662, 0.5489812188031324, 0.4033183338783049],
            "rotation": [90, 0, 0],
            "size":
            0.26588869520764724
        }, {
            "asset":
            "football",
            "position":
            [-0.3900378037115135, 0.8267650265556323, 0.4148765651537514],
            "rotation": [90, 0, 0],
            "size":
            0.2890051577585402
        }, {
            "asset":
            "basketball",
            "position":
            [-0.18117858987528535, 0.19665202236078264, 0.35448618047047975],
            "rotation": [90, 0, 0],
            "size":
            0.168224388391997
        }, {
            "asset": "table",
            "position": [0, 0.75, -0.06662601372551874],
            "rotation": [90, 0, 0],
            "size": 2
        }]
    }

    # 生成XML文件（输出路径可自定义）
    create_mujoco_xml(config=object_config, output_path="scene.xml")
