from blenderrenderer import Renderer
# import os
# os.chdir(r"E:\About_coding\fly_hand\scene_assets\textures")
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
render = Renderer(
    [0, 4, 3],
    60,
    [0, 0, 0], 
    has_wall=True,
    plane_texture=r"E:\About_coding\fly_hand\scene_assets\textures\rustic_floor.png",
    wall_texture=r"E:\About_coding\fly_hand\scene_assets\textures\gray_wall.png"
)
for obj in object_config["objects"]:
    render._setup_object(
        asset_path=asset_paths[obj["asset"]],
        size=obj["size"],
    )
positions = [obj["position"] for obj in object_config["objects"]]
rotations = [obj["rotation"] for obj in object_config["objects"]]
render.render(positions, rotations, "test.png")
