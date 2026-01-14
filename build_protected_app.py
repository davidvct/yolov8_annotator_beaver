import os
import shutil
import PyInstaller.__main__
import subprocess
import sys

def build():
    # 1. Clean previous builds
    print(" cleaning previous builds...")
    dirs_to_clean = ['build', 'dist', 'obf_dist']
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed {d}")

    # 2. Obfuscate with PyArmor
    print("Obfuscating code...")
    # Obfuscate utils and widgets packages to obf_dist
    # We use --recursive to handle subdirectories if any, output to obf_dist
    # We need to specify the packages. 
    # pyarmor gen -O obf_dist -r utils widgets
    
    cmd = ['pyarmor', 'gen', '-O', 'obf_dist', '-r', 'utils', 'widgets']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("PyArmor failed!")
        print(result.stderr)
        return
    print(result.stdout)

    # 3. Copy other necessary files
    print("Copying files to staging area...")
    files_to_copy = [
        'main.py',
        'build_info.py',
        'version_info.py',
        'environment.yml',
        'export_model_to_onnx.py'
    ]
    
    dirs_to_copy = [
        'ui',
        'models',
        'data_for_test'
    ]

    for f in files_to_copy:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join('obf_dist', f))
            print(f"Copied {f}")
            
    for d in dirs_to_copy:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join('obf_dist', d))
            print(f"Copied directory {d}")

    # Generate build info if not exists (or refresh it)
    import datetime
    now = datetime.datetime.now()
    date_str = now.strftime("%Y %b %d")
    with open(os.path.join("obf_dist", "build_info.py"), "w") as f:
        f.write('"""Auto-generated build info"""\n')
        f.write(f'BUILD_DATE = "{date_str}"\n')

    # 4. Run PyInstaller
    print("Running PyInstaller...")
    
    # We need to include the pyarmor_runtime which is generated in obf_dist
    # It acts as a package, so we should ensure it's picked up.
    # Changing CWD to obf_dist might be easier for PyInstaller to find things relative to main.py
    
    cwd = os.getcwd()
    os.chdir('obf_dist')
    

    # Helper to collect hidden imports
    def get_hidden_imports(package_dir):
        imports = []
        for root, _, files in os.walk(package_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    # Convert file path to module path
                    rel_path = os.path.relpath(os.path.join(root, file), start='.')
                    module_name = rel_path.replace(os.sep, '.')[:-3]
                    imports.append(f'--hidden-import={module_name}')
        return imports

    hidden_imports = []
    hidden_imports.extend(get_hidden_imports('utils'))
    hidden_imports.extend(get_hidden_imports('widgets'))
    
    print(f"Adding hidden imports: {hidden_imports}")


    # Helper to extract app name from main.py
    def get_app_name():
        import re
        try:
            with open('main.py', 'r') as f:
                content = f.read()
                match = re.search(r'app\.setApplicationName\((["\'])(.*?)\1\)', content)
                if match:
                    # Sanitize for filename (optional, but good practice to keep valid chars)
                    # For now we keep it as is, but spaces are allowed in Windows exe
                    name = match.group(2)
                    print(f"Found app name: {name}")
                    return name
        except Exception as e:
            print(f"Could not extract app name: {e}")
        return "YOLOv8Annotator" # Default fallback

    app_name = get_app_name()

    try:
        args = [
            'main.py',
            f'--name={app_name}',
            '--onedir',
            '--windowed',
            '--clean',
            '--noconfirm',
            '--collect-all=ultralytics',
            '--collect-all=PySide6',
            '--hidden-import=pyarmor_runtime', # Important for the obfuscated code
            '--distpath=../dist', # maintain original dist location
            '--workpath=../build', # maintain original build location
            '--specpath=..'       # maintain original spec location
        ] + hidden_imports
        
        PyInstaller.__main__.run(args)
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    build()
