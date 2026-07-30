from pathlib import Path

project_structure = {
    "data": ["raw", "processed"],
    "notebooks": [],
    "src": [
        "__init__.py",
        "preprocessing.py",
        "feature_engineering.py",
        "train.py",
        "predict.py"
    ],
    "models": [],
    "reports": [],
    "app": [],
    "tests": [],
    "configs": []
}

base = Path.cwd()

for folder, files in project_structure.items():
    folder_path = base / folder
    folder_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        (folder_path / file).touch()

(base / "README.md").touch()
(base / "requirements.txt").touch()
(base / ".gitignore").touch()

print("✅ Project structure created successfully!")