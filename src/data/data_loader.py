import os
from typing import List, Dict

class PDFFolderDataLoader:
    """
    Recursively loads PDF file paths and their category labels from nested folders.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_filepaths_and_labels(self) -> List[Dict[str, str]]:
        documents = []
        # Traverse each part folder
        for part_name in os.listdir(self.data_dir):
            part_path = os.path.join(self.data_dir, part_name)
            if os.path.isdir(part_path):
                # Traverse each category folder inside part
                for category in os.listdir(part_path):
                    category_path = os.path.join(part_path, category)
                    if os.path.isdir(category_path):
                        for filename in os.listdir(category_path):
                            if filename.lower().endswith('.pdf'):
                                filepath = os.path.join(category_path, filename)
                                # Document label = category folder name
                                documents.append({'filepath': filepath, 'label': category})
        return documents

if __name__ == "__main__":
    # Adjust to your actual path relative to src/data/data_loader.py
    # For example, if 'full_contract_pdf' isn't under 'data/raw', use the correct relative/absolute path.
    data_dir = r"C:\Users\Lenovo\legal-document-classifier\data\raw\full_contract_pdf"  # Use your path
    loader = PDFFolderDataLoader(data_dir=data_dir)
    docs = loader.load_filepaths_and_labels()
    print(f"Loaded {len(docs)} documents.")
    for d in docs[:5]:  # Print a few samples for sanity check
        print(d)
