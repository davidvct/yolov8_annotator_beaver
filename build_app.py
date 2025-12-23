import PyInstaller.__main__
import os

def build():
    PyInstaller.__main__.run([
        'main.py',
        '--name=YOLOv8Annotator',
        '--onedir',
        '--windowed',
        '--clean',
        '--noconfirm',
        # Collect everything from ultralytics to be safe
        '--collect-all=ultralytics', 
    ])

if __name__ == '__main__':
    build()
