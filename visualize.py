import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from fusion import feature_fusion
import visualization.clip_utils as clip_utils

import os
import sys
sys.path.append(".")

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    def __init__(self):
        
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        self.material = self._materials[Settings.LIT]


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21



    def __init__(self, width, height):
        self.pcd = None
        self.pcd_name = None

        self.settings = Settings()

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        # get the input text
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        
        tedit = gui.TextEdit()
        tedit.placeholder_text = "Edit me some text here"
        tedit.set_on_text_changed(self._on_text_changed)
        tedit.set_on_value_changed(self._on_value_changed)

        self._settings_panel.add_child(tedit)
        
        # set the layout
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        


        if gui.Application.instance.menubar is None:
            
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            
            menu = gui.Menu()

            menu.add_menu("File", file_menu)
            gui.Application.instance.menubar = menu


        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)

    
    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _on_text_changed(self, new_text):
        print(new_text)

    def _on_value_changed(self, new_text):
        if self.pcd_name is None:
            print("Please choose a .ply point cloud file")
        
        print()
        print("input:", new_text)
        print("===> Call Function to change the color of Point-Cloud <===")
        # self.load_cloud("replica_room0.ply")
        print()
        
        # reload the colors of points so the relavant points have distinctive color
        mask = clip_utils.find_mask(input, self.processed_mask3d, self.instance_feature, "")
        self.update_color(mask)
        self.update_scene()
        

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load_cloud(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def load_cloud(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        cloud = None
        
        try:
            cloud = o3d.io.read_point_cloud(path)
            self.pcd = cloud
            self.pcd_name = path
        except Exception as e:
            pass
        
        if cloud is not None:
            print("[Info] Successfully read", path)
            if not cloud.has_normals():
                cloud.estimate_normals()
            cloud.normalize_normals()
            geometry = cloud
        else:
            print("[WARNING] Failed to read points", path)

        if geometry is not None :
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                                   self.settings.material)
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def update_color(self, mask):
        color = np.asarray(self.pcd.colors)
        
        # @Ying change the new_color according to the mask 
        mask = np.random.random(color.shape[0]) # dummy mask:  value from 0 to 1
        new_color = clip_utils.generate_color_map(mask)

        assert(new_color.shape==color.shape)
        self.pcd.colors = o3d.utility.Vector3dVector(new_color)

    def update_scene(self):
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", self.pcd,
                                                   self.settings.material)

    # download the image to local 
    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main():
    ###################################################################
    # load processed instance mask
    processed_mask3d = np.loadtxt("0568/processed_scene0568_00_heatmap.txt") #(200, 232453) np.ndarray min 0, max 1
    print("read instance mask, done")


    # load per-point clip feature
    pc_clip = torch.load('0568/scene0568_00_0.pt')
    pc_features = pc_clip['feat'].numpy()           # (214318, 768) min:-2.1  max 2.3
    mask_ = pc_clip['mask_full'].numpy()             # (232453,)  [True, .. False]
    
    mask = np.asarray([mask_] * 200)                            # (200, 232453)
    processed_mask3d = processed_mask3d[mask] 
    processed_mask3d = np.reshape(processed_mask3d, [200, -1]) # (200, 214318)  
    print("read clip features, done")
    


    # now we have (200, 214318) "processed_mask3d" [0, 1] and (214318, 768) per point CLIP "pc_features" (-2.16, 2.37)

    # per instance clip feature (200, 768)
    feature_fusion(processed_mask3d, pc_features, "0568_00")
    instance_feature = np.loadtxt('test_data/fused_feature_0568_00.txt') # (200, 768) [-1.1, 1.2]
    print("compute instance CLIP features, done")
    ###################################################################
    
    gui.Application.instance.initialize()
    w = AppWindow(1024, 768)
    w.pc_features = pc_features
    w.processed_mask3d = processed_mask3d
    w.instance_feature = instance_feature

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load_cloud(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    gui.Application.instance.run()


if __name__ == "__main__":
    main()
