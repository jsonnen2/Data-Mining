
############ WRITTEN BY CHAT GPT ##############

import os
import ast

def get_dependencies(folder_path):
    dependencies = set()
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    dependencies.add(alias.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    dependencies.add(node.module.split('.')[0])
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
    return sorted(dependencies)

folder_path = "C:\J\Jobs\Data-Mining"
dependencies = get_dependencies(folder_path)
print("Dependencies:", dependencies)
