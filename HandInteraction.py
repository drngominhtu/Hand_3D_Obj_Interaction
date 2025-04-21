import cv2
import mediapipe as mp
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import trimesh
import pygame
from pygame.locals import *

class Model3D:
    def __init__(self, model_path):
        # Load the 3D model using trimesh with material support
        try:
            # Load with material support
            self.scene = trimesh.load(model_path, 
                                    force='mesh',
                                    process=False,  # Disable processing to preserve materials
                                    maintain_order=True,
                                    skip_materials=False)  # Ensure materials are loaded
            
            print(f"Loading model from: {model_path}")
            
            if isinstance(self.scene, trimesh.Scene):
                self.meshes = list(self.scene.geometry.values())
            else:
                self.meshes = [self.scene]
            
            # Debug materials
            print("\nMaterial Information:")
            for i, mesh in enumerate(self.meshes):
                print(f"\nMesh {i}:")
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                    mat = mesh.visual.material
                    print(f"- Has material: {type(mat)}")
                    if hasattr(mat, 'ambient'):
                        print(f"- Ambient: {mat.ambient}")
                    if hasattr(mat, 'diffuse'):
                        print(f"- Diffuse: {mat.diffuse}")
                    if hasattr(mat, 'specular'):
                        print(f"- Specular: {mat.specular}")
                else:
                    print("- No material found")
                    
            # Scale all meshes
            self.scale_model()
            self.zoom = 4.0  # Default zoom level
            self.rotation_x = 0
            self.rotation_y = 0
            
            # Load textures
            self.textures = {}
            self.load_textures()
            
            # Add color map for different parts
            self.mesh_colors = {
                0: {  # First mesh
                    'ambient': [0.0, 0.25, 0.0, 0.9],  # Dark green
                    'diffuse': [0.0, 0.4, 0.0, 0.9],
                    'specular': [0.2, 0.8, 0.2, 0.9],
                    'base_color': [0.0, 0.3, 0.0, 0.9]
                },
                1: {  # Second mesh
                    'ambient': [0.4, 0.0, 0.0, 0.9],  # Dark red
                    'diffuse': [0.6, 0.0, 0.0, 0.9],
                    'specular': [0.8, 0.2, 0.2, 0.9],
                    'base_color': [0.5, 0.0, 0.0, 0.9]
                },
                2: {  # Third mesh
                    'ambient': [0.0, 0.0, 0.3, 0.9],  # Dark blue
                    'diffuse': [0.0, 0.0, 0.5, 0.9],
                    'specular': [0.2, 0.2, 0.8, 0.9],
                    'base_color': [0.0, 0.0, 0.4, 0.9]
                }
                # Add more colors for more meshes if needed
            }
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def scale_model(self):
        # Get the bounding box of the model
        bbox = self.scene.bounds
        # Calculate the size
        size = bbox[1] - bbox[0]
        # Get the maximum dimension
        max_dim = max(size)
        # Calculate scale factor to make max dimension = 2.0
        scale = 2.0 / max_dim
        # Center the model
        center = (bbox[1] + bbox[0]) / 2
        # Transform vertices
        for mesh in self.meshes:
            mesh.vertices = (mesh.vertices - center) * scale
        
    def load_textures(self):
        for mesh in self.meshes:
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                if hasattr(mesh.visual.material, 'image'):
                    try:
                        # Convert PIL Image to numpy array
                        img = mesh.visual.material.image
                        img_array = np.array(img)
                        
                        # Create OpenGL texture
                        texture_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        
                        # Check image dimensions
                        if len(img_array.shape) == 3:  # Color image
                            if img_array.shape[2] == 4:  # RGBA
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_array.shape[1], img_array.shape[0], 
                                           0, GL_RGBA, GL_UNSIGNED_BYTE, img_array)
                            elif img_array.shape[2] == 3:  # RGB
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_array.shape[1], img_array.shape[0], 
                                           0, GL_RGB, GL_UNSIGNED_BYTE, img_array)
                        elif len(img_array.shape) == 2:  # Grayscale image
                            # Convert to RGB by repeating the grayscale values
                            rgb_array = np.stack((img_array,)*3, axis=-1)
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_array.shape[1], rgb_array.shape[0], 
                                       0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array)
                        
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                        self.textures[mesh] = texture_id
                    except Exception as e:
                        print(f"Warning: Failed to load texture: {e}")
                        continue

    def init_gl(self):
        # Print OpenGL information
        print("OpenGL Version:", glGetString(GL_VERSION).decode())
        print("OpenGL Renderer:", glGetString(GL_RENDERER).decode())
        print("OpenGL Vendor:", glGetString(GL_VENDOR).decode())
        print("OpenGL Shading Language Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Basic lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 10, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        
        # Enable color material for basic color support
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Màu mặc định (xám nhạt)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (800/600), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluLookAt(0, 2, self.zoom,
                  0, 0, 0,
                  0, 1, 0)
        
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        glEnable(GL_TEXTURE_2D)
        
        for i, mesh in enumerate(self.meshes):
            glPushMatrix()
            
            # Get colors for this mesh
            colors = self.mesh_colors.get(i, {
                'ambient': [0.2, 0.2, 0.2, 0.9],  # Default gray if no color specified
                'diffuse': [0.4, 0.4, 0.4, 0.9],
                'specular': [0.6, 0.6, 0.6, 0.9],
                'base_color': [0.3, 0.3, 0.3, 0.9]
            })
            
            # Apply colors
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, colors['ambient'])
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors['diffuse'])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, colors['specular'])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 75.0)
            
            # Set base color
            glColor4f(*colors['base_color'])
            
            # Draw mesh
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                vertices = mesh.vertices[face]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                normal = normal / np.linalg.norm(normal)
                
                for i, vertex in enumerate(vertices):
                    glNormal3fv(normal)
                    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                        glTexCoord2fv(mesh.visual.uv[face[i]])
                    glVertex3fv(vertex)
            glEnd()
            
            glPopMatrix()
        
        glDisable(GL_TEXTURE_2D)
        pygame.display.flip()

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.previous_x = None
        self.previous_y = None
        
        # Thêm danh sách để lưu trữ các giá trị chuyển động
        self.movement_history = []
        self.history_size = 5  # Kích thước lịch sử để tính trung bình

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def get_hand_direction(self, img):
        h, w, _ = img.shape
        horizontal_direction = "No hand detected"
        vertical_direction = "No hand detected"
        movement = {'x': 0, 'y': 0}  # Add movement values
        
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            hand_x = int(hand_landmarks.landmark[0].x * w)
            hand_y = int(hand_landmarks.landmark[0].y * h)
            
            if self.previous_x is not None and self.previous_y is not None:
                # Calculate movement
                movement['x'] = hand_x - self.previous_x
                movement['y'] = hand_y - self.previous_y
                
                # Lưu trữ các giá trị chuyển động vào lịch sử
                self.movement_history.append((movement['x'], movement['y']))
                if len(self.movement_history) > self.history_size:
                    self.movement_history.pop(0)  # Giữ kích thước lịch sử cố định

                # Tính toán giá trị trung bình
                avg_movement_x = sum(x for x, y in self.movement_history) / len(self.movement_history)
                avg_movement_y = sum(y for x, y in self.movement_history) / len(self.movement_history)

                movement['x'] = avg_movement_x
                movement['y'] = avg_movement_y
                
                # Horizontal movement
                if hand_x > self.previous_x + 5:
                    horizontal_direction = "Moving RIGHT"
                elif hand_x < self.previous_x - 5:
                    horizontal_direction = "Moving LEFT"
                else:
                    horizontal_direction = "Stationary"
                    
                # Vertical movement
                if hand_y > self.previous_y + 5:
                    vertical_direction = "Moving DOWN"
                elif hand_y < self.previous_y - 5:
                    vertical_direction = "Moving UP"
                else:
                    vertical_direction = "Stationary"
            
            self.previous_x = hand_x
            self.previous_y = hand_y
            
        return horizontal_direction, vertical_direction, movement

def main():
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (1200, 800)  # Kích thước cửa sổ chính
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Model Viewer")
    
    # Initialize the 3D model
    model = Model3D('d:/Code_Progress/HandInteraction/models/modelA/Food.obj')
    model.init_gl()
    
    # Thay đổi mức độ zoom để phóng to mô hình
    model.zoom = 4.0  # Giảm giá trị này để phóng to mô hình hơn
    
    # Initialize hand detection
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        
        # Clear the OpenGL buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render the 3D model
        model.render()
        
        # Draw the video stream in a smaller frame at the bottom left corner
        video_height, video_width, _ = img.shape
        frame_width = 100  # Giảm kích thước khung video xuống một nửa
        frame_height = int(video_height * (frame_width / video_width))  # Duy trì tỷ lệ khung hình
        
        # Set the viewport for the video stream
        glViewport(0, 0, frame_width, frame_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, frame_width, frame_height, 0, -1, 1)  # Set orthographic projection for video frame
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Draw the video stream
        glDrawPixels(video_width, video_height, GL_BGR, GL_UNSIGNED_BYTE, img)
        
        # Reset viewport to the full window for the 3D model
        glViewport(0, 0, display[0], display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Update model rotation based on hand movement
        horizontal_dir, vertical_dir, movement = detector.get_hand_direction(img)
        
        # Debugging: Print movement values
        print(f"Movement: {movement}")  # Check if movement values are being updated
        
        model.rotation_y += movement['x'] * 0.9  # Adjust rotation speed as needed
        model.rotation_x += movement['y'] * 0.9  # Adjust rotation speed as needed
        
        # Display the hand tracking window
        cv2.imshow("Hand Detection", img)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                return
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
